from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class HeadMixture(nn.Module):
    def __init__(self, n_matrices: int, n_heads: int, tau: float = 1.0, l2_delta: float = 1e-4):
        super().__init__()
        self.n_matrices = n_matrices
        self.n_heads = n_heads
        self.tau = float(tau)
        self.l2_delta = float(l2_delta)
        self.alpha = nn.Parameter(torch.zeros(n_matrices))
        self.delta = nn.Parameter(torch.zeros(n_heads, n_matrices))

    def mixture_weights(self, layer_index: int | None = None) -> torch.Tensor:
        del layer_index
        # Centering removes softmax's arbitrary additive offsets and improves identifiability.
        alpha = self.alpha - self.alpha.mean()
        delta = self.delta - self.delta.mean(dim=-1, keepdim=True)
        return torch.softmax((alpha.unsqueeze(0) + delta) / self.tau, dim=-1)

    def regularization(self) -> torch.Tensor:
        delta_centered = self.delta - self.delta.mean(dim=-1, keepdim=True)
        return self.l2_delta * delta_centered.square().mean()

    def forward(
        self,
        matrices: torch.Tensor,
        layer_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del layer_index
        _validate_matrices(matrices, self.n_matrices)
        pi_float = self.mixture_weights()
        pi = pi_float.to(device=matrices.device, dtype=matrices.dtype)
        bias = torch.einsum("hm,bmij->bhij", pi, matrices)
        return bias, pi_float, self.regularization()


class LayerwiseHeadMixture(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_matrices: int,
        n_heads: int,
        tau: float = 1.0,
        l2_logits: float = 0.0,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_matrices = int(n_matrices)
        self.n_heads = int(n_heads)
        self.tau = float(tau)
        self.l2_logits = float(l2_logits)
        self.logits = nn.Parameter(torch.empty(n_layers, n_heads, n_matrices))
        if init_std > 0:
            nn.init.normal_(self.logits, mean=0.0, std=float(init_std))
        else:
            nn.init.zeros_(self.logits)

    def mixture_weights(self, layer_index: int | None = None) -> torch.Tensor:
        centered = self.logits - self.logits.mean(dim=-1, keepdim=True)
        weights = torch.softmax(centered / self.tau, dim=-1)
        if layer_index is None:
            return weights
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer_index={layer_index} outside [0,{self.n_layers})")
        return weights[layer_index]

    def regularization(self) -> torch.Tensor:
        centered = self.logits - self.logits.mean(dim=-1, keepdim=True)
        return self.l2_logits * centered.square().mean()

    def forward(
        self,
        matrices: torch.Tensor,
        layer_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_index is None:
            raise ValueError("LayerwiseHeadMixture requires layer_index")
        _validate_matrices(matrices, self.n_matrices)
        pi_float = self.mixture_weights(layer_index)
        pi = pi_float.to(device=matrices.device, dtype=matrices.dtype)
        bias = torch.einsum("hm,bmij->bhij", pi, matrices)
        return bias, pi_float, self.regularization()


def _validate_matrices(matrices: torch.Tensor, n_matrices: int) -> None:
    if matrices.ndim != 4:
        raise ValueError(f"Expected matrices [B,M,T,T], got {tuple(matrices.shape)}")
    if matrices.shape[1] != n_matrices:
        raise ValueError(f"Expected {n_matrices} matrices, received {matrices.shape[1]}")


class GraphBiasedSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_graph_bias = config.use_graph_bias

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.beta = nn.Parameter(torch.full((config.n_head,), float(config.beta_init)))

    @staticmethod
    def _masked_zscore(bias: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        """Differentiable normalization over valid residue-residue cells."""
        original_dtype = bias.dtype
        work = bias.float()
        valid = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]
        if work.shape[1] != 1:
            valid = valid.expand(-1, work.shape[1], -1, -1)
        count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        masked = torch.where(valid, work, torch.zeros_like(work))
        mean = masked.sum(dim=(-2, -1), keepdim=True) / count
        variance = torch.where(valid, (work - mean).square(), torch.zeros_like(work)).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        normalized = (work - mean) / variance.sqrt().clamp_min(1e-6)
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        return normalized.to(original_dtype)

    def forward(
        self,
        x: torch.Tensor,
        graph_bias: torch.Tensor | None,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_graph_bias and graph_bias is not None:
            normalized_bias = self._masked_zscore(graph_bias, valid_tokens)
            attention = attention + self.beta[None, :, None, None].to(attention.dtype) * normalized_bias.to(
                attention.dtype
            )

        key_mask = valid_tokens[:, None, None, :]
        attention = attention.masked_fill(~key_mask, torch.finfo(attention.dtype).min)
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        y = torch.matmul(attention, v)
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y * valid_tokens.unsqueeze(-1).to(y.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GraphBiasedSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        graph_bias: torch.Tensor | None,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), graph_bias=graph_bias, valid_tokens=valid_tokens)
        x = x + self.mlp(self.ln_2(x))
        return x * valid_tokens.unsqueeze(-1).to(x.dtype)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.empty(n_embd))
        self.norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scores = torch.einsum("btd,d->bt", x, self.query) / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(~valid_tokens, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.einsum("bt,btd->bd", weights, x)


class PhaseekV3Classifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        if config.graph_mixer == "shared":
            self.mixer: HeadMixture | LayerwiseHeadMixture = HeadMixture(
                n_matrices=config.topk_m,
                n_heads=config.n_head,
                tau=config.mixture_tau,
                l2_delta=config.mixture_l2,
            )
        else:
            self.mixer = LayerwiseHeadMixture(
                n_layers=config.n_layer,
                n_matrices=config.topk_m,
                n_heads=config.n_head,
                tau=config.mixture_tau,
                l2_logits=config.mixture_l2,
                init_std=config.mixture_init_std,
            )
        self.pooler = (
            MaskedAttentionPooling(config.n_embd, config.resid_pdrop)
            if config.pooling == "attention"
            else None
        )
        self.classifier_dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.n_embd, 2)

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, matrices: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, Any]]:
        batch, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError(f"Input length {length} exceeds block_size {self.config.block_size}")
        valid_tokens = tokens.ne(0)
        if not torch.all(valid_tokens.any(dim=1)):
            raise ValueError("Every sequence must contain at least one non-padding token")

        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        x = x * valid_tokens.unsqueeze(-1).to(x.dtype)

        mixture_weights = self.mixer.mixture_weights()
        mixture_regularization = self.mixer.regularization()

        if self.config.graph_mixer == "shared":
            if matrices is not None:
                graph_bias, _, _ = self.mixer(matrices)
            else:
                graph_bias = None
            for block in self.blocks:
                x = block(x, graph_bias=graph_bias, valid_tokens=valid_tokens)
        else:
            for layer_index, block in enumerate(self.blocks):
                if matrices is not None:
                    graph_bias, _, _ = self.mixer(matrices, layer_index=layer_index)
                else:
                    graph_bias = None
                x = block(x, graph_bias=graph_bias, valid_tokens=valid_tokens)

        x = self.final_norm(x)

        if self.pooler is not None:
            pooled = self.pooler(x, valid_tokens)
        else:
            mask = valid_tokens.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(self.classifier_dropout(pooled))
        auxiliary = {
            "mixture_weights": mixture_weights,
            "mixture_regularization": mixture_regularization,
        }
        return logits, auxiliary

    @staticmethod
    def _is_graph_parameter(name: str) -> bool:
        return name.startswith("mixer.") or name.endswith("attn.beta")

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze sequence backbone while leaving graph path and classification head trainable."""
        for name, parameter in self.named_parameters():
            keep_trainable = (
                self._is_graph_parameter(name)
                or name.startswith("classifier")
                or name.startswith("pooler")
            )
            parameter.requires_grad_(not frozen or keep_trainable)

    def graph_parameter_items(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if self._is_graph_parameter(name)
        ]

    def optimizer_groups(
        self,
        weight_decay: float,
        base_lr: float | None = None,
        graph_lr_multiplier: float = 1.0,
    ) -> list[dict[str, Any]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        graph: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if self._is_graph_parameter(name):
                graph.append(parameter)
                continue
            excluded = (
                parameter.ndim < 2
                or name.endswith(".bias")
                or "embedding" in name
                or "norm" in name
                or name == "pooler.query"
            )
            if excluded:
                no_decay.append(parameter)
            else:
                decay.append(parameter)

        assigned = {id(p) for p in decay + no_decay + graph}
        expected = {id(p) for p in self.parameters()}
        if assigned != expected:
            raise RuntimeError("Optimizer parameter grouping is incomplete or duplicated")

        groups: list[dict[str, Any]] = [
            {"params": decay, "weight_decay": weight_decay, "group_name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
            {"params": graph, "weight_decay": 0.0, "group_name": "graph"},
        ]
        if base_lr is not None:
            groups[0]["lr"] = float(base_lr)
            groups[1]["lr"] = float(base_lr)
            groups[2]["lr"] = float(base_lr) * float(graph_lr_multiplier)
        return groups
