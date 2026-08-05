from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedGroupKFold

from .matrices import inspect_npz
from .tokenizer import normalize_sequence, validate_sequence
from .utils import sequence_hash


def read_fasta(path: str | Path, label: int) -> pd.DataFrame:
    records = []
    seen: set[str] = set()
    for record in SeqIO.parse(str(path), "fasta"):
        sample_id = str(record.id)
        if sample_id in seen:
            raise ValueError(f"Duplicate FASTA identifier {sample_id!r} in {path}")
        seen.add(sample_id)
        sequence = normalize_sequence(str(record.seq))
        validate_sequence(sequence, sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "description": str(record.description),
                "sequence": sequence,
                "length": len(sequence),
                "label": int(label),
                "sequence_hash": sequence_hash(sequence),
            }
        )
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame(records)


def graph_index_from_directory(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("*.npz")):
        metadata = inspect_npz(path)
        sample_id = metadata["sample_id"] or path.stem
        rows.append(
            {
                "sample_id": str(sample_id),
                "npz_path": str(path.resolve()),
                "npz_sequence_length": metadata["sequence_length"],
                "matrix_count": metadata["matrix_count"],
                "matrix_shapes": repr(metadata["matrix_shapes"]),
            }
        )
    if not rows:
        raise ValueError(f"No NPZ files found in {directory}")
    frame = pd.DataFrame(rows)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate graph sample IDs in {directory}: {duplicates[:10]}")
    return frame


def read_graph_index(source: str | Path) -> pd.DataFrame:
    source = Path(source)
    if source.is_dir():
        return graph_index_from_directory(source)
    frame = pd.read_csv(source)
    required = {"sample_id", "npz_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Graph index {source} missing columns: {sorted(missing)}")
    frame = frame.copy()
    base = source.parent
    frame["npz_path"] = frame["npz_path"].map(
        lambda value: str((base / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    return frame


def _best_group_holdout(
    frame: pd.DataFrame,
    target_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["group_id"].astype(str).to_numpy()
    n_groups = frame["group_id"].nunique()
    if n_groups < 3:
        raise ValueError("At least three distinct groups are required for train/val/test splitting")

    overall_rate = float(y.mean())
    ideal_splits = max(2, round(1.0 / target_fraction))
    candidates = sorted(
        set(
            n
            for n in range(max(2, ideal_splits - 2), min(n_groups, ideal_splits + 2) + 1)
            if n >= 2
        )
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    X_dummy = np.zeros((len(frame), 1), dtype=np.float32)

    for offset in range(12):
        for n_splits in candidates:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=seed + offset,
                )
                for train_idx, holdout_idx in splitter.split(X_dummy, y, groups):
                    train_y = y[train_idx]
                    holdout_y = y[holdout_idx]
                    if len(np.unique(train_y)) < 2 or len(np.unique(holdout_y)) < 2:
                        continue
                    fraction = len(holdout_idx) / len(frame)
                    score = (
                        4.0 * abs(fraction - target_fraction)
                        + abs(float(holdout_y.mean()) - overall_rate)
                        + 0.5 * abs(float(train_y.mean()) - overall_rate)
                    )
                    if best is None or score < best[0]:
                        best = (score, train_idx, holdout_idx)
            except ValueError:
                continue

    if best is None:
        raise ValueError(
            "Could not construct a grouped split containing both classes. "
            "Check group_id values and class counts."
        )
    return best[1], best[2]


def assign_grouped_splits(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-8):
        raise ValueError("Split fractions must sum to 1")
    remaining_idx, test_idx = _best_group_holdout(frame, test_fraction, seed)
    remaining = frame.iloc[remaining_idx].reset_index().rename(columns={"index": "original_index"})
    relative_val = val_fraction / (train_fraction + val_fraction)
    train_local, val_local = _best_group_holdout(remaining, relative_val, seed + 1000)

    split = np.full(len(frame), "", dtype=object)
    split[test_idx] = "test"
    split[remaining.iloc[train_local]["original_index"].to_numpy(dtype=int)] = "train"
    split[remaining.iloc[val_local]["original_index"].to_numpy(dtype=int)] = "val"
    if np.any(split == ""):
        raise RuntimeError("Internal split assignment error")

    result = frame.copy()
    result["split"] = split
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(result.loc[result.split == a, "group_id"]) & set(
            result.loc[result.split == b, "group_id"]
        )
        if overlap:
            raise RuntimeError(f"Group leakage between {a} and {b}: {list(overlap)[:5]}")
    for name in ("train", "val", "test"):
        labels = set(result.loc[result.split == name, "label"])
        if labels != {0, 1}:
            raise ValueError(f"Split {name!r} does not contain both classes")
    return result


def build_manifest(
    pos_fasta: str | Path,
    neg_fasta: str | Path,
    pos_graphs: str | Path,
    neg_graphs: str | Path,
    groups_csv: str | Path | None,
    topk_m: int,
    max_length: int,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    sequences = pd.concat(
        [read_fasta(pos_fasta, 1), read_fasta(neg_fasta, 0)],
        ignore_index=True,
    )
    if sequences["sample_id"].duplicated().any():
        duplicates = sequences.loc[sequences["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Sample IDs must be globally unique across classes: {duplicates[:10]}")

    conflicting = sequences.groupby("sequence_hash")["label"].nunique()
    conflicts = conflicting[conflicting > 1]
    if len(conflicts):
        raise ValueError(
            f"Found {len(conflicts)} exact sequences carrying conflicting labels. Resolve these before training."
        )

    graphs = pd.concat(
        [read_graph_index(pos_graphs), read_graph_index(neg_graphs)],
        ignore_index=True,
    )
    if graphs["sample_id"].duplicated().any():
        duplicates = graphs.loc[graphs["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Graph index contains duplicate sample IDs: {duplicates[:10]}")

    manifest = sequences.merge(graphs, on="sample_id", how="left", validate="one_to_one")
    missing_graphs = manifest.loc[manifest["npz_path"].isna(), "sample_id"].tolist()
    if missing_graphs:
        raise ValueError(
            f"No graph file matched {len(missing_graphs)} FASTA records. Examples: {missing_graphs[:10]}. "
            "Matching is by exact sample_id; positional matching is intentionally forbidden."
        )
    extra_graphs = sorted(set(graphs["sample_id"]) - set(sequences["sample_id"]))
    if extra_graphs:
        raise ValueError(
            f"Found {len(extra_graphs)} graph files with no FASTA record. Examples: {extra_graphs[:10]}"
        )

    if (manifest["length"] > max_length).any():
        examples = manifest.loc[manifest["length"] > max_length, ["sample_id", "length"]].head(10)
        raise ValueError(
            f"{int((manifest['length'] > max_length).sum())} sequences exceed max_length={max_length}. "
            f"Examples:\n{examples.to_string(index=False)}"
        )
    if "npz_sequence_length" in manifest:
        mismatched = manifest[
            manifest["npz_sequence_length"].notna()
            & (manifest["npz_sequence_length"].astype(int) != manifest["length"])
        ]
        if len(mismatched):
            raise ValueError(
                "Sequence-length metadata mismatch between FASTA and NPZ. Examples:\n"
                + mismatched[["sample_id", "length", "npz_sequence_length"]].head(10).to_string(index=False)
            )
    if "matrix_count" in manifest and (manifest["matrix_count"] < topk_m).any():
        examples = manifest.loc[manifest["matrix_count"] < topk_m, ["sample_id", "matrix_count"]].head(10)
        raise ValueError(
            f"Some NPZ files contain fewer than topk_m={topk_m} matrices. Examples:\n"
            + examples.to_string(index=False)
        )

    if groups_csv is not None:
        groups = pd.read_csv(groups_csv)
        if not {"sample_id", "group_id"}.issubset(groups.columns):
            raise ValueError("groups CSV must contain sample_id and group_id")
        manifest = manifest.merge(groups[["sample_id", "group_id"]], on="sample_id", how="left", validate="one_to_one")
        if manifest["group_id"].isna().any():
            missing = manifest.loc[manifest["group_id"].isna(), "sample_id"].head(10).tolist()
            raise ValueError(f"Missing group_id values for samples: {missing}")
    else:
        # Exact duplicates are kept together. For mutation/augmentation datasets, provide parent IDs explicitly.
        manifest["group_id"] = "seqsha:" + manifest["sequence_hash"]

    manifest = assign_grouped_splits(
        manifest,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    report = summarize_manifest(manifest, topk_m=topk_m, max_length=max_length, groups_supplied=groups_csv is not None)
    return manifest, report


def summarize_manifest(frame: pd.DataFrame, topk_m: int, max_length: int, groups_supplied: bool) -> dict:
    split_counts = (
        frame.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "negative", 1: "positive"})
    )
    train = frame[frame.split == "train"]
    n_neg = int((train.label == 0).sum())
    n_pos = int((train.label == 1).sum())
    imbalance = max(n_neg, n_pos) / max(1, min(n_neg, n_pos))
    lengths = frame["length"].to_numpy()
    return {
        "total_samples": int(len(frame)),
        "total_positive": int((frame.label == 1).sum()),
        "total_negative": int((frame.label == 0).sum()),
        "split_counts": split_counts.to_dict(orient="index"),
        "train_imbalance_ratio": float(imbalance),
        "length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
        },
        "distinct_groups": int(frame.group_id.nunique()),
        "groups_supplied": bool(groups_supplied),
        "topk_m": int(topk_m),
        "max_length": int(max_length),
        "warning": None
        if groups_supplied
        else (
            "group_id was derived from exact sequence hashes. If samples are mutations or augmentations of a parent "
            "protein, rebuild with --groups-csv so all relatives remain in the same split."
        ),
    }
