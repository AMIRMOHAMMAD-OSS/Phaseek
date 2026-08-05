import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from Configue import CfgNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chars = "ACDEFGHIKLMNPQRSTVWY"
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "../model/Trained_BPE2.json")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.model_max_length = 256


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class NY(nn.Module):
    def forward(self, x):
        return 3 * torch.tanh(0.3 * x)


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
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
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class ClassifierI(nn.Module):
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = len(chars)
        C.max_length = 512
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        self.max_length = config.block_size
        self.soft = nn.Softmax(dim=1)
        self.config = self.get_default_config()
        self.device = device
        self.model_states = {
            "h": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
            "g": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "f": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
            "e": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
            "d": {"n_layer": 8, "n_head": 16, "n_embd": 512},
            "c": {"n_layer": 6, "n_head": 6, "n_embd": 192},
            "b": {"n_layer": 4, "n_head": 4, "n_embd": 128},
            "a": {"n_layer": 3, "n_head": 3, "n_embd": 48},
        }

        if config.model_type is not None:
            config.merge_from_dict(self.model_states[config.model_type])

        self.ny = NY()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.max_length, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.classifier_head = nn.Sequential(nn.Linear(config.n_embd, 2))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @torch.no_grad()
    def predict_proba(self, idx, j="None"):
        self.eval()
        x, _ = self.forward(idx)
        if j == "None":
            return x[:, 0:1]
        elif j == "soft":
            return self.soft(x)[:, 0:1]
        else:
            return torch.sigmoid(x)[:, 0:1]

    def forward(self, idx, targets=None):
        device_ = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device_).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.ny(self.classifier_head(x)).mean(1).to(device_)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss


class transformer:
    def __init__(self, mode):
        self.mode = mode
        self.device = device
        self.vocab_dict = tokenizer.get_vocab()
        self._classifier = self.create_classifier()

    def create_classifier(self):
        model_config = ClassifierI.get_default_config()
        model_config.vocab_size = 25
        model_config.block_size = 512

        if self.mode == "b":
            model_config.model_type = "b"
            model = ClassifierI(model_config)
            state = torch.load(
                os.path.join(
                    os.path.dirname(__file__), "../model/Core_model_860k_10_FEGS_features"
                ),
                map_location=self.device,
            )
            model.load_state_dict(state)
        else:
            model_config.model_type = "c"
            model = ClassifierI(model_config)
            state = torch.load(
                os.path.join(
                    os.path.dirname(__file__), "../model/Core_model_1_5M_10_FEGS_features.pt"
                ),
                map_location=self.device,
            )
            model.load_state_dict(state)

        model.to(self.device)
        return model

    def Encode(self, seq_list):
        def encode_char(s):
            return [self.vocab_dict.get(ch, 0) for ch in s]

        PAD = 0
        max_len = 512
        encoded = list(map(encode_char, seq_list))
        padded = np.array(
            [seq[:max_len] + [PAD] * (max_len - len(seq)) for seq in encoded],
            dtype=np.int64,
        )
        return torch.tensor(padded, device=self.device, dtype=torch.long)

    def Decode(self, arr):
        def decode_row(row):
            ids = [j for j in row if j != 0]
            out = []
            for j in ids:
                if 1 <= j <= len(chars):
                    out.append(chars[j - 1])
            return "".join(out)

        H = list(map(decode_row, arr))
        G = []
        u = 0
        for i in H:
            if len(i) > 512:
                u = H.index(i)
                for j in range(len(i) - 512):
                    G.append(i[j : j + 512])
            else:
                G.append(i)
        return G, u

    def enhancer(self, x):
        x = np.asarray(x)
        vals = x[:, 0] if (x.ndim == 2 and x.shape[1] == 1) else x.ravel()
        wei = np.array([np.exp((v - 1) / (v + 1e-12)) for v in vals])
        return float(np.sum(vals * wei) / np.sum(wei))

    @torch.no_grad()
    def predict_proba(self, seqs):
        if len(seqs[0]) > 512:
            windows = [
                self.Encode([seqs[0][j : j + 512]]) for j in range(len(seqs[0]) - 512)
            ]
            if len(windows) > 700:
                chunks = [
                    torch.cat(windows[i * 700 : (i + 1) * 700])
                    for i in range(len(windows) // 700)
                ]
                if len(windows) % 700:
                    chunks.append(torch.cat(windows[(len(windows) // 700) * 700 :]))
                preds = [
                    self._classifier.predict_proba(t, "sig").cpu().numpy()
                    for t in chunks
                ]
                preds = np.concatenate(preds, axis=0)
                return self.enhancer(preds)
            else:
                batch = torch.cat(windows)
                preds = (
                    self._classifier.predict_proba(batch, "sig").cpu().numpy()
                )
                return self.enhancer(preds)
        else:
            batch = self.Encode(seqs)
            preds = self._classifier.predict_proba(batch, "sig").cpu().numpy()
            return preds.reshape(len(seqs), 1)
