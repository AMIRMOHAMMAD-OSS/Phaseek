#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from phaseek_v2.fegs_fast import (
    FIXED_SELECTION_METHOD,
    FastFEGSExtractor,
)

_EXTRACTOR = None
_MOTIF_INDICES = np.arange(10, dtype=np.int32)
_OUT_DIR = Path(".")
_OVERWRITE = False


def parse_motif_indices(value: str) -> np.ndarray:
    try:
        indices = np.asarray(
            [int(item.strip()) for item in value.split(",") if item.strip()],
            dtype=np.int32,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--motif-indices must be comma-separated integers"
        ) from exc
    if len(indices) != 10:
        raise argparse.ArgumentTypeError(
            f"Exactly 10 motif indices are required, received {indices.tolist()}"
        )
    if len(np.unique(indices)) != len(indices):
        raise argparse.ArgumentTypeError("Motif indices must be unique")
    return indices


def safe_filename(sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode()).hexdigest()[:12]
    return f"{slug[:90]}__{digest}.npz"


def init_worker(m_mat, motif_indices, out_dir, overwrite):
    global _EXTRACTOR, _MOTIF_INDICES, _OUT_DIR, _OVERWRITE
    _EXTRACTOR = FastFEGSExtractor(m_mat)
    _MOTIF_INDICES = np.asarray(motif_indices, dtype=np.int32)
    _OUT_DIR = Path(out_dir)
    _OVERWRITE = overwrite


def scalar(value):
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def cached_file_is_valid(path: Path, sample_id: str, expected_hash: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(
                [k for k in z.files if k.startswith("M") and k[1:].isdigit()],
                key=lambda k: int(k[1:]),
            )
            stored_indices = np.asarray(z["motif_indices"], dtype=np.int32)
            return (
                scalar(z["sample_id"]) == sample_id
                and scalar(z["sequence_sha256"]) == expected_hash
                and len(keys) == len(_MOTIF_INDICES)
                and np.array_equal(stored_indices, _MOTIF_INDICES)
                and scalar(z["selection_method"]) == FIXED_SELECTION_METHOD
                and int(np.asarray(z["format_version"]).reshape(-1)[0]) >= 3
            )
    except Exception:
        return False


def process_row(row):
    sample_id, sequence, expected_hash = row
    path = _OUT_DIR / safe_filename(sample_id)

    if path.exists() and not _OVERWRITE:
        if cached_file_is_valid(path, sample_id, expected_hash):
            return {
                "sample_id": sample_id,
                "npz_path": str(path.resolve()),
                "status": "cached",
                "bytes": path.stat().st_size,
            }

    result = _EXTRACTOR.extract_selected_graphs(
        sample_id,
        sequence,
        motif_indices=_MOTIF_INDICES,
    )
    payload = {
        f"M{i}": matrix.astype(np.float16, copy=False)
        for i, matrix in enumerate(result.matrices)
    }
    payload.update(
        sample_id=np.asarray(sample_id),
        sequence_length=np.asarray(len(sequence), dtype=np.int32),
        sequence_sha256=np.asarray(expected_hash),
        motif_indices=result.motif_indices.astype(np.int32),
        motif_orderings=result.motif_orderings,
        shap_rank=np.arange(1, len(result.matrices) + 1, dtype=np.int32),
        selection_method=np.asarray(result.selection_method),
        storage_dtype=np.asarray("float16"),
        format_version=np.asarray(3, dtype=np.int32),
    )

    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)

    return {
        "sample_id": sample_id,
        "npz_path": str(path.resolve()),
        "status": "written",
        "bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--m-mat", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument(
        "--motif-indices",
        type=parse_motif_indices,
        default=parse_motif_indices("0,1,2,3,4,5,6,7,8,9"),
        help="Fixed SHAP-ranked M.mat rows; default: 0,1,...,9",
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_id = (
        args.task_id
        if args.task_id is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )

    dataframe = pd.read_csv(args.manifest)
    required = {"sample_id", "sequence", "sequence_sha256"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}")

    start = task_id * args.chunk_size
    stop = min(len(dataframe), start + args.chunk_size)
    if start >= len(dataframe):
        print(f"No rows for task {task_id}")
        return

    rows = [
        (str(row.sample_id), str(row.sequence), str(row.sequence_sha256))
        for row in dataframe.iloc[start:stop].itertuples(index=False)
    ]

    output_dir = Path(args.out_dir)
    index_dir = Path(args.index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Fixed motif indices:",
        args.motif_indices.tolist(),
        "selection_method:",
        FIXED_SELECTION_METHOD,
        flush=True,
    )

    if args.processes <= 1:
        init_worker(
            args.m_mat,
            args.motif_indices,
            str(output_dir),
            args.overwrite,
        )
        results = [
            process_row(row)
            for row in tqdm(rows, desc=f"task {task_id}")
        ]
    else:
        with Pool(
            args.processes,
            initializer=init_worker,
            initargs=(
                args.m_mat,
                args.motif_indices,
                str(output_dir),
                args.overwrite,
            ),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_row, rows, chunksize=1),
                    total=len(rows),
                    desc=f"task {task_id}",
                )
            )

    part = pd.DataFrame(results).sort_values("sample_id")
    part["task_id"] = task_id
    part["row_start"] = start
    part["row_stop"] = stop

    target = index_dir / f"graph_index_part_{task_id:04d}.csv"
    temporary = target.with_suffix(".tmp.csv")
    part.to_csv(temporary, index=False)
    os.replace(temporary, target)

    print(part.status.value_counts().to_string())
    print(
        f"Rows {start}:{stop}; bytes={int(part.bytes.sum()):,}; index={target}"
    )


if __name__ == "__main__":
    main()
