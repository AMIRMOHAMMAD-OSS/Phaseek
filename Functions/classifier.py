from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tokenizers import Tokenizer
from Configue import CfgNode


if torch.cuda.is_available() and False:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

chars = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = Tokenizer.from_file("../Trained_BPE2.json")
tokenizer.model_max_length = 256



import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class SiLU(nn.Module):
   def forward(self, x):
        return x*F.sigmoid(x)

class NY(nn.Module):
  def forward(self,x):
    return 3*torch.tanh(0.3*x)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
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
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        C.vocab_size = len(chars)
        C.max_length = 512
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        self.max_length = config.block_size
        self.soft = nn.Softmax(1)
        self.config = self.get_default_config()
        self.device = device
        self.model_states = {'h':{'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
                                    'g':{'n_layer': 12, '': 12, 'n_embd': 768},
                'f':   {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
                'e':   {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
                'd':{'n_layer': 8, 'n_head': 16, 'n_embd': 512},
                'c':{'n_layer': 6, 'n_head': 6, 'n_embd': 192},
                'b':{'n_layer': 4, 'n_head': 4, 'n_embd': 128},
                'a':{'n_layer': 3, 'n_head': 3, 'n_embd': 48}}

        type_ = config.model_type is not None
        #assert type_ in "abcdefgh"
        p = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        #assert type_ == True and p == True
        if type_:
            config.merge_from_dict(self.model_states[config.model_type])
        self.closs = nn.BCELoss()
        self.ny = NY()
        #self.l = nn.Linear(512,1,64)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_length, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),))
        self.classifier_head = nn.Sequential(#nn.Tanh(),
                                             nn.Linear(config.n_embd, 2)
                                             )
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        #print("[ Number of trainable parameters: %.2fM ]" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 25
        config.max_length = 512
        model = ClassifierI(config)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!"                                                     % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @torch.no_grad
    def predict_proba(self,idx,j ="None"):
      sigmoid = nn.Sigmoid()
      soft = nn.Softmax(dim=1)
      ny = NY()
      si = SiLU()
      self.eval()
      x,_ = self.forward(idx)
      if j == "None":
        return x[:,0:1]
      elif j == "soft":
        return self.soft(x)[:,0:1]
      else:
        return sigmoid(x)[:,0:1]

    @torch.no_grad
    def predict(self,idx):
      g = torch.zeros((idx.shape[0],1))
      e = -1
      for i in self.predict_proba(idx):
        e+=1
        if i[0].item()>=0.5:
          g[e] = 1
      return g

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.ny(self.classifier_head(x)).mean(1).to(device)
        #logits = self.l(logits)
        # = F.sigmoid(logits.view(b,2).mean(1).view(b,1))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits,targets,ignore_index = -1)
        return logits, loss
        



class Transformer():
    def __init__(self, mode):
          self.mode = mode
          self.device = device  # Use the passed device
          self.vocab_dict = tokenizer.get_vocab()

    def classifier(self):
          device = self.device  # Use the instance's device attribute
          model_config = ClassifierI.get_default_config()
          model_config.vocab_size = 25
          model_config.block_size = 512
          if self.mode == "b":
              model_config.model_type = 'b'
              model2 = ClassifierI(model_config)
              model2.load_state_dict(torch.load("../model_b", map_location=self.device, weights_only=True))
          else:
              model_config.model_type = 'c'
              model2 = ClassifierI(model_config)
              model2.load_state_dict(torch.load("../model_c", map_location=self.device, weights_only=True))
          model2.to(device)
          return model2

    def Encode(self, i):
          def encode_char(x):
              return [self.vocab_dict[char] for char in x]

          def pad_sequences(x, PAD=0, max_len=512):
              return np.array([seq + [PAD] * (max_len - len(seq)) for seq in x])

          encoded = list(map(encode_char, i))
          padded = pad_sequences(encoded)
          return torch.tensor(padded).to(self.device)

    def Decode(self,i):
      def decode(k):
        l = [j for j in k if j != 0 ]
        seq = [chars[j] for j in l]
        return "".join(seq)
      H = list(map(decode,i))
      G = []
      for i in H:
        if len(i)>512:
          u = H.index(i)
          for j in range(len(i)-512):
            G.append(i[j:j+512])
        else:
          G.append(i)
      return G,u
    def Mean(self,i):
      I = list(i.ravel())
      sig = lambda x:np.e**(0*x)
      Sum = [sig(np.abs(x-int(len(I)/2))) for x in range(len(I))]
      wei = np.array([Sum[i]/sum(Sum) for i in range(len(I))]).reshape(len(I),1)
      return np.sum(wei*i)

    def enhancer(self,x):
      wei = np.array([np.exp((i[0]-1)/(i[0])) for i in x])
      return np.sum(np.array([i[0] for i in x])*wei)/np.sum(wei)

    def predict_proba(self,i):
      if len(i[0])>512:
        l = [self.Encode([i[0][j:j+512]]) for j in range(len(i[0])-512)]
        if len(l)>700:
          L = [torch.concat(tuple(l[i*700:(i+1)*700])) for i in range(len(l)//700)] + [torch.concat(tuple(l[(len(l)//700)*700:(len(l)//700)*700+len(l)%700]))]
          T = [self.classifier().predict_proba(i,"sig").tolist() for i in L ]
          return self.enhancer(np.concatenate(tuple([i for i in T])))
        else:
          t = torch.concat(tuple(l))
          return self.enhancer(self.classifier().predict_proba(t,"sig").tolist())
      else:
        seq = self.Encode(i)
        U = np.array(self.classifier().predict_proba(seq,"sig").tolist())
        out = np.array(U)
        return out.reshape(len(i),1)
