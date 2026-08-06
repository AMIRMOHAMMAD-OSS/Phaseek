from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .matrices import MatrixLoadConfig, MatrixStore
from .tokenizer import encode_sequence
from .utils import seed_worker

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "sequence",
    "label",
    "group_id",
    "npz_path",
    "split",
}


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["sequence"] = frame["sequence"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["npz_path"] = frame["npz_path"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id values in manifest, examples: {duplicates}")
    if not set(frame["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("split must contain only train, val, or test")
    for npz_path in frame["npz_path"]:
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"Missing graph file: {npz_path}")
    return frame


class PhaseekDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_length: int):
        self.rows = frame.reset_index(drop=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        tokenized = encode_sequence(
            row.sequence,
            max_length=self.max_length,
            sample_id=row.sample_id,
        )
        return {
            "sample_id": row.sample_id,
            "tokens": tokenized.tokens,
            "length": tokenized.true_length,
            "unknown_count": tokenized.unknown_count,
            "label": int(row.label),
            "npz_path": row.npz_path,
        }


@dataclass
class GraphCollator:
    matrix_config: MatrixLoadConfig

    def __post_init__(self) -> None:
        self.store: MatrixStore | None = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.store is None:
            self.store = MatrixStore(self.matrix_config)
        tokens = np.stack([item["tokens"] for item in batch], axis=0)
        labels = np.asarray([item["label"] for item in batch], dtype=np.int64)
        matrices = np.stack(
            [
                self.store.load(
                    item["npz_path"],
                    sample_id=item["sample_id"],
                    sequence_length=item["length"],
                )
                for item in batch
            ],
            axis=0,
        )
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "tokens": torch.from_numpy(tokens).long(),
            "labels": torch.from_numpy(labels).long(),
            "matrices": torch.from_numpy(matrices),
            "unknown_counts": torch.tensor([item["unknown_count"] for item in batch], dtype=torch.long),
        }


def make_loader(
    frame: pd.DataFrame,
    max_length: int,
    topk_m: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    cache_items: int,
    matrix_dtype: str,
    strict_matrices: bool,
    shuffle: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    dataset = PhaseekDataset(frame, max_length=max_length)
    collator = GraphCollator(
        MatrixLoadConfig(
            target_length=max_length,
            topk_m=topk_m,
            output_dtype=matrix_dtype,
            strict=strict_matrices,
            cache_items=cache_items,
        )
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": collator,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
