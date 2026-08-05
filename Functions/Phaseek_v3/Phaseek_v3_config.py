from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 22
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    topk_m: int = 10
    embd_pdrop: float = 0.10
    resid_pdrop: float = 0.10
    attn_pdrop: float = 0.10
    pooling: str = "attention"
    graph_mixer: str = "shared"
    mixture_tau: float = 1.0
    mixture_l2: float = 1e-4
    mixture_init_std: float = 0.0
    beta_init: float = 0.01
    use_graph_bias: bool = True

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        if self.graph_mixer not in {"shared", "layerwise"}:
            raise ValueError("graph_mixer must be 'shared' or 'layerwise'")
        if self.topk_m < 1:
            raise ValueError("topk_m must be positive")
        if self.mixture_tau <= 0:
            raise ValueError("mixture_tau must be positive")
        if self.mixture_l2 < 0:
            raise ValueError("mixture_l2 cannot be negative")
        if self.mixture_init_std < 0:
            raise ValueError("mixture_init_std cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 42
    deterministic: bool = False
    epochs: int = 0
    patience: int = 8
    batch_size: int = 0
    effective_batch_size: int = 0
    num_workers: int = 8
    prefetch_factor: int = 3
    matrix_cache_items: int = 32
    learning_rate: float = 0.0
    graph_lr_multiplier: float = 1.0
    freeze_backbone_epochs: int = 0
    min_learning_rate_ratio: float = 0.05
    weight_decay: float = 0.01
    warmup_fraction: float = 0.08
    label_smoothing: float = 0.02
    max_grad_norm: float = 1.0
    balance_mode: str = "auto"
    amp: str = "auto"
    selection_metric: str = "pr_auc"
    strict_matrices: bool = True
    matrix_dtype: str = "float16"
    compile_model: bool = False

    def validate(self) -> None:
        if self.balance_mode not in {"auto", "none", "weighted"}:
            raise ValueError("balance_mode must be auto, none, or weighted")
        if self.selection_metric not in {"pr_auc", "roc_auc", "loss"}:
            raise ValueError("selection_metric must be pr_auc, roc_auc, or loss")
        if self.amp not in {"auto", "bf16", "fp16", "none"}:
            raise ValueError("amp must be auto, bf16, fp16, or none")
        if self.matrix_dtype not in {"float16", "float32"}:
            raise ValueError("matrix_dtype must be float16 or float32")
        if self.graph_lr_multiplier <= 0:
            raise ValueError("graph_lr_multiplier must be positive")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs cannot be negative")
        if self.freeze_backbone_epochs >= self.epochs:
            raise ValueError("freeze_backbone_epochs must be smaller than epochs")

    def to_dict(self) -> dict:
        return asdict(self)


def choose_model_profile(n_train: int, requested: str = "auto") -> dict:
    if requested not in {"auto", "small", "base"}:
        raise ValueError("profile must be auto, small, or base")
    if requested == "auto":
        requested = "small" if n_train < 1800 else "base"
    if requested == "small":
        return {"profile": "small", "n_layer": 4, "n_head": 4, "n_embd": 128}
    return {"profile": "base", "n_layer": 6, "n_head": 6, "n_embd": 192}


def resolve_training_defaults(
    config: TrainConfig,
    n_train: int,
    gpu_memory_gb: float,
) -> TrainConfig:
    resolved = TrainConfig(**config.to_dict())

    if resolved.batch_size <= 0:
        if gpu_memory_gb >= 70:
            resolved.batch_size = 32
        elif gpu_memory_gb >= 38:
            resolved.batch_size = 24
        elif gpu_memory_gb >= 22:
            resolved.batch_size = 16
        elif gpu_memory_gb > 0:
            resolved.batch_size = 8
        else:
            resolved.batch_size = 4

    if resolved.effective_batch_size <= 0:
        resolved.effective_batch_size = 64 if n_train < 2000 else 128
    resolved.effective_batch_size = max(resolved.batch_size, resolved.effective_batch_size)

    if resolved.epochs <= 0:
        resolved.epochs = 55 if n_train < 1800 else (40 if n_train < 10000 else 28)

    if resolved.learning_rate <= 0:
        scale = math.sqrt(resolved.effective_batch_size / 64.0)
        resolved.learning_rate = min(5e-4, 3e-4 * scale)

    resolved.validate()
    return resolved


def class_weights_from_counts(n_negative: int, n_positive: int, mode: str = "auto") -> tuple[float, float] | None:
    if n_negative <= 0 or n_positive <= 0:
        raise ValueError("Training split must contain both classes")
    imbalance = max(n_negative, n_positive) / min(n_negative, n_positive)
    if mode == "none" or (mode == "auto" and imbalance < 1.25):
        return None
    total = n_negative + n_positive
    return total / (2.0 * n_negative), total / (2.0 * n_positive)
