import os
import time
import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from Configue import CfgNode
from FEGS_features_extraction import FEGSFeatureExtractor

AA20 = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {aa: i for i, aa in enumerate(AA20)}

CHARS = "ACDEFGHIKLMNPQRSTVWY"
ID2AA = {i + 1: ch for i, ch in enumerate(CHARS)} 
ID2AA[0] = "X"


def build_pair_bias_from_FEGS_SAD(seq: str) -> np.ndarray:
    AAC, DPC = FEGSFeatureExtractor._SAD_static((seq, AA20))
    T = len(seq)
    idxs = np.array([AA_INDEX.get(ch, 0) for ch in seq], dtype=int)
    P = DPC[idxs][:, idxs].astype(np.float32)  
    P = 0.5 * (P + P.T)
    return P


def ids_to_sequence_str(ids_row: np.ndarray) -> str:
    letters = []
    for v in ids_row:
        v = int(v)
        if v in ID2AA:
            aa = ID2AA[v]
            if aa != "X":
                letters.append(aa)
    return "".join(letters)


class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.device = 'cuda'
        C.num_workers = 4
        C.max_iters = 200         
        C.epochs = 20
        C.batch_size = 128
        C.max_length = 512
        C.learning_rate = 8e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        C.use_bias = False      
        C.log_every = 50
        C.save_path = "./model_fegs.pt"
        return C

    def __init__(self, config, model, train_dataset, val_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def _balanced_sample(self, data_np: np.ndarray, batch_size: int):

        labels = data_np[:, -1].astype(int)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            start = np.random.randint(0, max(1, len(data_np) - batch_size))
            batch = data_np[start:start + batch_size]
            seq = batch[:, :-1].astype(np.int64)
            y = batch[:, -1].astype(np.int64)
            return seq, y

        N_pos = np.random.randint(0, batch_size + 1)
        N_neg = batch_size - N_pos

        pos_sel = np.random.choice(pos_idx, size=min(N_pos, len(pos_idx)), replace=len(pos_idx) < N_pos)
        neg_sel = np.random.choice(neg_idx, size=min(N_neg, len(neg_idx)), replace=len(neg_idx) < N_neg)
        sel = np.concatenate([pos_sel, neg_sel], axis=0)
        np.random.shuffle(sel)

        batch = data_np[sel]
        seq = batch[:, :-1].astype(np.int64) 
        y = batch[:, -1].astype(np.int64)     
        return seq, y

    def _random_slice(self, data_np: np.ndarray, batch_size: int):
        start = np.random.randint(0, max(1, len(data_np) - batch_size))
        batch = data_np[start:start + batch_size]
        seq = batch[:, :-1].astype(np.int64)
        y = batch[:, -1].astype(np.int64)
        return seq, y

    def _maybe_build_bias_batch(self, seq_ids_batch: torch.Tensor) -> torch.Tensor | None:
        if not self.config.use_bias:
            return None

        seq_ids_np = seq_ids_batch.detach().cpu().numpy() 
        B, T = seq_ids_np.shape
        out = np.zeros((B, T, T), dtype=np.float32)

        for b in range(B):
            seq_str = ids_to_sequence_str(seq_ids_np[b])
            P = build_pair_bias_from_FEGS_SAD(seq_str)
            if P.shape[0] < T:
                P_pad = np.zeros((T, T), dtype=np.float32)
                P_pad[:P.shape[0], :P.shape[0]] = P
                out[b] = P_pad
            else:
                out[b] = P[:T, :T]

        return torch.tensor(out, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _eval_split(self, data_np: np.ndarray, batches: int = 200):
        self.model.eval()
        losses = []

        for _ in range(batches):
            seq_np, y_np = self._random_slice(data_np, self.config.batch_size)
            X = torch.tensor(seq_np, dtype=torch.long, device=self.device)
            Y = torch.tensor(y_np, dtype=torch.long, device=self.device)   
            bias = self._maybe_build_bias_batch(X) 
            logits, loss = self.model(X, targets=Y, bias_matrix=bias)
            losses.append(loss.item())

        self.model.train()
        return float(np.mean(losses))
    def run(self):
        cfg = self.config

        print(f"[trainer] device: {self.device}")
        print(f"[trainer] epochs={cfg.epochs} iters/epoch={cfg.max_iters} batch_size={cfg.batch_size}")
        print(f"[trainer] lr={cfg.learning_rate} betas={cfg.betas} wd={cfg.weight_decay}")
        print(f"[trainer] use_bias={cfg.use_bias}")

        for epoch in range(cfg.epochs):
            t0 = time.time()
            running = []

            for it in range(cfg.max_iters):
                seq_np, y_np = self._balanced_sample(self.train_dataset, cfg.batch_size)
                X = torch.tensor(seq_np, dtype=torch.long, device=self.device)  
                Y = torch.tensor(y_np, dtype=torch.long, device=self.device)    
                bias = self._maybe_build_bias_batch(X)
                logits, loss = self.model(X, targets=Y, bias_matrix=bias)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), cfg.grad_norm_clip)
                self.optimizer.step()

                running.append(loss.item())

                if (it + 1) % cfg.log_every == 0:
                    avg = float(np.mean(running[-cfg.log_every:]))
                    print(f"epoch {epoch+1:03d} | it {it+1:04d}/{cfg.max_iters} | loss {avg:.4f}")

            train_loss = self._eval_split(self.train_dataset, batches=100)
            val_loss = self._eval_split(self.val_dataset, batches=100)
            dt = time.time() - t0
            print(f"[epoch {epoch+1:03d}] train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")

            self.trigger_callbacks("epoch_end")

            try:
                torch.save(self.model.state_dict(), cfg.save_path)
                print(f"[saved] {cfg.save_path}")
            except Exception as e:
                print(f"[warn] could not save model: {e}")
