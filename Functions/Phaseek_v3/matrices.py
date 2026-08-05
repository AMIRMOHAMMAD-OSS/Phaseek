from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatrixLoadConfig:
    target_length: int
    topk_m: int
    output_dtype: str = "float16"
    strict: bool = True
    cache_items: int = 32


class LRUCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, int(max_items))
        self._store: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


def _decode_scalar(value: np.ndarray | str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _matrix_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    numbered = [key for key in npz.files if key.startswith("M") and key[1:].isdigit()]
    if numbered:
        return sorted(numbered, key=lambda key: int(key[1:]))
    return []


def inspect_npz(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        keys = _matrix_keys(npz)
        if keys:
            shapes = [tuple(np.asarray(npz[key]).shape) for key in keys]
            matrix_count = len(keys)
        elif "arr_0" in npz and np.asarray(npz["arr_0"]).ndim == 3:
            arr = np.asarray(npz["arr_0"])
            matrix_count = int(arr.shape[0])
            shapes = [tuple(arr.shape[1:])] * matrix_count
        else:
            candidates = [np.asarray(npz[key]) for key in npz.files]
            matrices = [arr for arr in candidates if arr.ndim == 2]
            matrix_count = len(matrices)
            shapes = [tuple(arr.shape) for arr in matrices]
        return {
            "path": str(path),
            "sample_id": _decode_scalar(npz["sample_id"]) if "sample_id" in npz else None,
            "sequence_length": int(np.asarray(npz["sequence_length"]).reshape(-1)[0])
            if "sequence_length" in npz
            else None,
            "matrix_count": matrix_count,
            "matrix_shapes": shapes,
        }


def _read_matrices(npz: np.lib.npyio.NpzFile) -> list[np.ndarray]:
    keys = _matrix_keys(npz)
    if keys:
        return [np.asarray(npz[key], dtype=np.float32) for key in keys]
    if "arr_0" in npz:
        arr = np.asarray(npz["arr_0"])
        if arr.ndim == 3:
            return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]
    matrices: list[np.ndarray] = []
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim == 2:
            matrices.append(np.asarray(arr, dtype=np.float32))
    if len(matrices) > 1:
        scores = [float(np.linalg.norm(matrix, ord="fro")) for matrix in matrices]
        matrices = [matrices[i] for i in np.argsort(scores)[::-1]]
    return matrices


def standardize_valid_matrix(matrix: np.ndarray, valid_length: int, target_length: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Graph matrix must be square, got shape={matrix.shape}")
    if matrix.shape[0] < valid_length:
        raise ValueError(
            f"Graph matrix side {matrix.shape[0]} is shorter than sequence length {valid_length}"
        )
    valid = np.asarray(matrix[:valid_length, :valid_length], dtype=np.float32)
    valid = np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(valid.mean(dtype=np.float64))
    std = float(valid.std(dtype=np.float64))
    normalized = np.zeros_like(valid, dtype=np.float32) if std < 1e-6 else (valid - mean) / std
    out = np.zeros((target_length, target_length), dtype=np.float32)
    out[:valid_length, :valid_length] = normalized
    return out


class MatrixStore:
    def __init__(self, config: MatrixLoadConfig):
        self.config = config
        self.cache = LRUCache(config.cache_items)

    def load(self, path: str | Path, sample_id: str, sequence_length: int) -> np.ndarray:
        path = str(Path(path).resolve())
        if sequence_length > self.config.target_length:
            raise ValueError(
                f"Sample {sample_id!r} length {sequence_length} exceeds target length "
                f"{self.config.target_length}; silent graph truncation is disabled."
            )
        cache_key = (
            path,
            sample_id,
            sequence_length,
            self.config.target_length,
            self.config.topk_m,
            self.config.output_dtype,
            self.config.strict,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        with np.load(path, allow_pickle=False) as npz:
            stored_id = _decode_scalar(npz["sample_id"]) if "sample_id" in npz else None
            if stored_id is not None and stored_id != sample_id:
                raise ValueError(
                    f"Sequence/graph ID mismatch: manifest={sample_id!r}, NPZ={stored_id!r}, path={path}"
                )
            if "sequence_length" in npz:
                stored_length = int(np.asarray(npz["sequence_length"]).reshape(-1)[0])
                if stored_length != sequence_length:
                    raise ValueError(
                        f"Sequence/graph length mismatch for {sample_id!r}: "
                        f"manifest={sequence_length}, NPZ={stored_length}"
                    )
            matrices = _read_matrices(npz)

        if len(matrices) < self.config.topk_m and self.config.strict:
            raise ValueError(
                f"{path} contains {len(matrices)} graph matrices, fewer than topk_m={self.config.topk_m}"
            )
        selected = matrices[: self.config.topk_m]
        while len(selected) < self.config.topk_m:
            selected.append(np.zeros((sequence_length, sequence_length), dtype=np.float32))

        stack = np.stack(
            [
                standardize_valid_matrix(matrix, sequence_length, self.config.target_length)
                for matrix in selected
            ],
            axis=0,
        )
        dtype = np.float16 if self.config.output_dtype == "float16" else np.float32
        stack = stack.astype(dtype, copy=False)
        self.cache.put(cache_key, stack)
        return stack
