from typing import Tuple
import numpy as np, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .utils import parse_complex_to_float_array

class SeqFeatDataset(Dataset):
    def __init__(self, seq_array: np.ndarray, feat_array: np.ndarray, labels: np.ndarray):
        assert seq_array.shape[0] == feat_array.shape[0] == labels.shape[0]
        self.seq  = seq_array.astype(np.int64,   copy=False)
        self.feat = feat_array.astype(np.float32, copy=False)
        self.lab  = labels.astype(np.int64,      copy=False)
    def __len__(self): return self.seq.shape[0]
    def __getitem__(self, i):
        return self.seq[i], self.feat[i], int(self.lab[i])

def collate_cpu(batch):
    seqs, feats, labs = zip(*batch)
    seqs  = torch.tensor(np.stack(seqs,  axis=0), dtype=torch.long)
    feats = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)
    labs  = torch.tensor(labs, dtype=torch.long)
    return seqs, feats, labs

def load_feature_tables(pos_csv: str, neg_csv: str, num_feat: int, pos_rows=None, neg_rows=None):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    if pos_rows is None: pos_rows = len(pos_df)
    if neg_rows is None: neg_rows = len(neg_df)
    pos_feats = parse_complex_to_float_array(pos_df.iloc[:pos_rows, :num_feat].values)
    neg_feats = parse_complex_to_float_array(neg_df.iloc[:neg_rows, :num_feat].values)
    return pos_feats, neg_feats

def make_loaders(
    Xseq_tr, Xseq_va, Xf_tr, Xf_va, y_tr, y_va,
    batch_size: int, num_workers: int, pin_memory: bool
) -> Tuple[DataLoader, DataLoader]:
    class_counts   = np.bincount(y_tr, minlength=2).astype(np.float64)
    sample_weights = np.array([1.0 / class_counts[c] for c in y_tr], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = SeqFeatDataset(Xseq_tr, Xf_tr, y_tr)
    val_ds   = SeqFeatDataset(Xseq_va, Xf_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_cpu)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_cpu)
    return train_loader, val_loader
