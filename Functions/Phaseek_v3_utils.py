import os, math, random, shutil
from glob import glob
from typing import List
import numpy as np
import torch
from collections import OrderedDict

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device

def copy_once(src: str, dst: str):
    try:
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Copying {src} -> {dst} (one-time)")
            shutil.copytree(src, dst)
        elif os.path.exists(dst):
            print(f"Using local cached folder: {dst}")
        else:
            print(f"[WARN] Source not found: {src}. Using it directly if accessible.")
    except Exception as e:
        print(f"[WARN] Skipped copy {src} -> {dst}: {e}")

def list_npz_sorted(dir_path: str) -> List[str]:
    paths = sorted(glob(os.path.join(dir_path, "*.npz")))
    assert len(paths) > 0, f"No .npz files found in {dir_path}"
    return paths

def true_len_ell(tokens_row: np.ndarray) -> int:
    return int((tokens_row != 0).sum())

def standardize_and_pad(M: np.ndarray, ell: int, T: int) -> np.ndarray:
    L = M.shape[0]
    mu = M.mean(dtype=np.float64)
    sd = M.std(dtype=np.float64)
    if sd < 1e-6: sd = 1e-6
    Z = (M - mu) / sd
    out = np.zeros((T, T), dtype=np.float32)
    L_use = min(L, T)
    out[:L_use, :L_use] = Z[:L_use, :L_use].astype(np.float32, copy=False)
    return out

class LRUCache:
    def __init__(self, max_items=256):
        self.max_items = max_items
        self.store = OrderedDict()
    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None
    def put(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.max_items:
            self.store.popitem(last=False)
