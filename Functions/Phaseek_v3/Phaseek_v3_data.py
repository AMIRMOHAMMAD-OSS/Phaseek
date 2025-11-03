from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .utils import true_len_ell, standardize_and_pad, LRUCache

TOPK_M = 10
SEQ_LEN = 512
FP16_BIAS = True

_lru = LRUCache(max_items=256)

def load_topk_unpadded(npz_path: str, ell: int, T: int, topk: int = TOPK_M) -> np.ndarray:
    import numpy as np
    with np.load(npz_path, allow_pickle=False) as z:
        mats = []
        M_keys = [k for k in z.files if k.startswith("M") and k[1:].isdigit()]
        if len(M_keys) > 0:
            M_keys = sorted(M_keys, key=lambda k: int(k[1:]))
            for k in M_keys[:topk]:
                M = np.asarray(z[k], dtype=np.float32)
                if M.ndim == 2:
                    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                    mats.append(M)
        else:
            if "arr_0" in z and z["arr_0"].ndim == 3:
                A = z["arr_0"]
                for i in range(min(A.shape[0], topk)):
                    M = np.asarray(A[i], dtype=np.float32)
                    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                    mats.append(M)
            else:
                for k in z.files:
                    M = np.asarray(z[k], dtype=np.float32)
                    if M.ndim == 2 and M.size > 0:
                        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                        mats.append(M)
                if len(mats) > topk:
                    scores = [float(np.linalg.norm(M, ord='fro')) for M in mats]
                    idx = np.argsort(scores)[::-1][:topk]
                    mats = [mats[i] for i in idx]
        while len(mats) < topk:
            mats.append(np.zeros((1,1), dtype=np.float32))
    stack = np.stack([standardize_and_pad(M, ell=ell, T=T) for M in mats[:topk]], axis=0)  # (m,T,T)
    return stack

def build_Lhat_stack_from_npz(npz_path: str, ell: int, T: int = SEQ_LEN, topk: int = TOPK_M) -> np.ndarray:
    key = (npz_path, T)
    cached = _lru.get(key)
    if cached is not None:
        return cached
    stack = load_topk_unpadded(npz_path, ell=ell, T=T, topk=topk)
    _lru.put(key, stack)
    return stack

class SeqTopkDataset(Dataset):
    def __init__(self, seq_array: np.ndarray, label_array: np.ndarray, npz_paths: np.ndarray):
        assert seq_array.shape[0] == label_array.shape[0] == npz_paths.shape[0]
        self.seq = seq_array.astype(np.int64, copy=False)
        self.lab = label_array.astype(np.int64, copy=False)
        self.paths = npz_paths.astype(str)
    def __len__(self): return self.seq.shape[0]
    def __getitem__(self, idx): return self.seq[idx], int(self.lab[idx]), self.paths[idx]

def collate_with_Lhat_cpu(batch):
    seqs, labs, paths = zip(*batch)
    seqs   = np.stack(seqs, axis=0)
    ells   = [true_len_ell(seqs[i]) for i in range(seqs.shape[0])]
    stacks = [build_Lhat_stack_from_npz(paths[i], ell=ells[i], T=SEQ_LEN, topk=TOPK_M) for i in range(len(paths))]
    Lhat   = torch.tensor(np.stack(stacks, axis=0), dtype=torch.float16 if FP16_BIAS else torch.float32)
    seqs_t = torch.tensor(seqs, dtype=torch.long)
    labels = torch.tensor(labs, dtype=torch.long)
    return seqs_t, labels, Lhat

def make_dataloaders(
    Xseq_tr: np.ndarray, Xseq_va: np.ndarray,
    paths_tr: np.ndarray, paths_va: np.ndarray,
    y_tr: np.ndarray, y_va: np.ndarray,
    batch_size: int, num_workers: int, prefetch: int, device: str
) -> Tuple[DataLoader, DataLoader]:
    PIN_MEMORY = (device == "cuda")

    train_ds = SeqTopkDataset(Xseq_tr, y_tr, paths_tr)
    val_ds   = SeqTopkDataset(Xseq_va, y_va, paths_va)

    class_counts   = np.bincount(y_tr, minlength=2).astype(np.float64)
    sample_weights = np.array([1.0 / class_counts[c] for c in y_tr], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch,
        collate_fn=collate_with_Lhat_cpu
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch,
        collate_fn=collate_with_Lhat_cpu
    )
    return train_loader, val_loader
