import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device

def parse_complex_to_float_array(arr_like) -> np.ndarray:
    """Robust parser used in your cell: complex/strings -> float32 with NaN/Inf -> 0."""
    arr = np.asarray(arr_like)
    out = np.empty(arr.shape, dtype=np.float32)
    it = np.nditer(arr, flags=['multi_index', 'refs_ok', 'zerosize_ok'], op_flags=['readonly'])
    for x in it:
        v = x.item()
        if isinstance(v, (np.floating, float, int)): val = float(v)
        elif isinstance(v, complex): val = float(v.real)
        elif isinstance(v, str):
            s = v.strip()
            try:
                val = complex(s).real if (s.startswith("(") and s.endswith(")")) else float(s)
            except Exception:
                val = 0.0
        else:
            try: val = float(v)
            except Exception: val = 0.0
        out[it.multi_index] = val
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
