import math, torch
import torch.nn as nn
import torch.nn.functional as F

class BiasBuilder(nn.Module):

    def __init__(self, T: int, dist_lambda: float = 64.0, outer_scale: float = 1.0,
                 rbf_scale: float = 0.0, rbf_sigma: float = 1.0):
        super().__init__()
        self.T = T
        self.dist_lambda = dist_lambda
        self.outer_scale = outer_scale
        self.rbf_scale   = rbf_scale
        self.rbf_sigma   = rbf_sigma
        idx = torch.arange(T).float()
        dist = (idx[:, None] - idx[None, :]).abs()
        self.register_buffer("dist_bias", - dist / dist_lambda)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.size(0)
        x = feats
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        xz = (x - mu) / sd                             
        w = F.interpolate(xz.unsqueeze(1), size=self.T, mode="linear", align_corners=True).squeeze(1)  
        w = (w - w.mean(dim=1, keepdim=True)) / (w.std(dim=1, keepdim=True).clamp_min(1e-6))
        outer = self.outer_scale * torch.einsum("bi,bj->bij", w, w)
        if self.rbf_scale > 0.0:
            diff = w.unsqueeze(2) - w.unsqueeze(1)
            rbf  = - (diff * diff) / (2.0 * (self.rbf_sigma ** 2))
            outer = outer + self.rbf_scale * rbf
        bias = outer + self.dist_bias
        mean = bias.mean(dim=(1,2), keepdim=True)
        std  = bias.std(dim=(1,2), keepdim=True).clamp_min(1e-6)
        return (bias - mean) / std

class FEGSTrans(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.causal = getattr(config, "causal", False)
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, bias_matrix=None, key_padding_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if bias_matrix is not None:
            bm = bias_matrix
            if bm.dim() == 3: bm = bm.unsqueeze(1)
            bm = bm.to(att.dtype)
            att = att + self.beta * bm

        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :T]
            att = att.masked_fill(~key_mask, -1e4)
        if self.causal:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e4)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = FEGSTrans(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x, bias_matrix=None, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), bias_matrix=bias_matrix, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Config:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)

class TransformerClassifier(nn.Module):
    def __init__(self, config: Config, num_feat: int, seq_len: int, label_smooth: float = 0.05):
        super().__init__()
        self.config = config
        self.label_smooth = label_smooth

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0),
            wpe  = nn.Embedding(config.block_size,  config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.head = nn.Linear(config.n_embd, 2)
        self.bias_builder = BiasBuilder(T=seq_len, dist_lambda=64.0, outer_scale=1.0, rbf_scale=0.0, rbf_sigma=1.0)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, feats_1d, targets=None):
        B, T = idx.size()
        dev = idx.device
        bias_matrix = self.bias_builder(feats_1d.to(dev))    
        pos = torch.arange(0, T, dtype=torch.long, device=dev).unsqueeze(0)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        key_padding_mask = (idx != 0)
        for block in self.transformer.h:
            x = block(x, bias_matrix=bias_matrix, key_padding_mask=key_padding_mask)
        x = self.transformer.ln_f(x)

        valid = key_padding_mask.float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        logits = self.head(pooled)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smooth)
        return logits, loss

    def configure_optimizers(self, lr, betas=(0.9, 0.95), weight_decay=0.1):
        decay, no_decay = set(), set()
        for name, module in self.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                full = f"{name}.{pname}" if name else pname
                if pname.endswith("bias"): no_decay.add(full)
                elif isinstance(module, (nn.LayerNorm,)): no_decay.add(full)
                else: decay.add(full)
        for emb in ["transformer.wte.weight", "transformer.wpe.weight"]:
            if emb in decay: decay.remove(emb); no_decay.add(emb)
        param_dict = {pn: p for pn in self.named_parameters()}
        return torch.optim.AdamW(
            [
                {"params": [param_dict[pn] for pn in sorted(decay)],    "weight_decay": weight_decay},
                {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=lr, betas=betas
        )
