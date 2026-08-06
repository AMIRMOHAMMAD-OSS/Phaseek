from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_torch() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {k: json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)


def resolve_amp_dtype(device: torch.device, requested: str = "auto") -> tuple[bool, torch.dtype | None, bool]:
    """Return (autocast_enabled, autocast_dtype, use_grad_scaler)."""
    if device.type != "cuda" or requested == "none":
        return False, None, False

    if requested == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        requested = "bf16" if major >= 8 else "fp16"

    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but is not supported by this CUDA device.")
        return True, torch.bfloat16, False
    if requested == "fp16":
        return True, torch.float16, True
    raise ValueError(f"Unknown AMP mode: {requested}")


def gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024**3)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
