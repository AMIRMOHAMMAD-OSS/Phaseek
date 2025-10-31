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
        self.use_graph_bias = getattr(config, "use_graph_bias", True)
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

        if self.use_graph_bias and (bias_matrix is not None):
            if bias_matrix.dim() == 3:
                bm = bias_matrix.unsqueeze(1) 
            elif bias_matrix.dim() == 4:
                bm = bias_matrix
            else:
                raise ValueError("bias_matrix must be (B,T,T) or (B,1,T,T)")
            bm = bm[:, :, :T, :T].to(att.dtype).to(att.device)
            mean = bm.mean(dim=(-2, -1), keepdim=True)
            std  = bm.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            bm = (bm - mean) / std
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
        y = self.resid_dropout(self.c_proj(y))
        return y

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
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TransformerClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0),
            wpe  = nn.Embedding(config.block_size,  config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.head = nn.Linear(config.n_embd, 2)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, bias_matrix=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1,T)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        key_padding_mask = (idx != 0)  
        for block in self.transformer.h:
            x = block(x, bias_matrix=bias_matrix, key_padding_mask=key_padding_mask)
        x = self.transformer.ln_f(x)
        valid = key_padding_mask.float().unsqueeze(-1)  # (B,T,1)
        x_sum = (x * valid).sum(dim=1)                  # (B,C)
        lens  = valid.sum(dim=1).clamp_min(1.0)         # (B,1)
        pooled = x_sum / lens

        logits = self.head(pooled)  # (B,2)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=0.05)
        return logits, loss
    def configure_optimizers(self, lr, betas=(0.9, 0.95), weight_decay=0.1):
        decay, no_decay = set(), set()
        for name, module in self.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                full = f"{name}.{pname}" if name else pname
                if pname.endswith("bias"):
                    no_decay.add(full)
                elif isinstance(module, (nn.LayerNorm,)):
                    no_decay.add(full)
                else:
                    decay.add(full)
        for emb in ["transformer.wte.weight", "transformer.wpe.weight"]:
            if emb in decay:
                decay.remove(emb)
                no_decay.add(emb)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],    "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
