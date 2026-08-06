#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phaseek_v3.fegs_fast import FIXED_SELECTION_METHOD


def scalar(value):
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def parse_motif_indices(value: str) -> np.ndarray:
    indices = np.asarray(
        [int(item.strip()) for item in value.split(",") if item.strip()],
        dtype=np.int32,
    )
    if len(indices) != 10 or len(np.unique(indices)) != 10:
        raise argparse.ArgumentTypeError(
            "Exactly ten unique comma-separated motif indices are required"
        )
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--motif-indices",
        type=parse_motif_indices,
        default=parse_motif_indices("0,1,2,3,4,5,6,7,8,9"),
    )
    args = parser.parse_args()

    sequences = pd.read_csv(args.sequence_manifest)
    parts = sorted(Path(args.index_dir).glob("graph_index_part_*.csv"))
    if not parts:
        raise FileNotFoundError(f"No index parts in {args.index_dir}")

    graphs = (
        pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
        .drop_duplicates("sample_id", keep="last")
    )
    merged = sequences.merge(
        graphs[["sample_id", "npz_path"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    missing = merged[merged.npz_path.isna()]
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} samples missing matrices: "
            f"{missing.sample_id.head(10).tolist()}"
        )

    errors: list[str] = []
    total_bytes = 0

    for row in merged.itertuples(index=False):
        path = Path(row.npz_path)
        if not path.is_file():
            errors.append(f"missing:{row.sample_id}")
            continue
        total_bytes += path.stat().st_size

        try:
            with np.load(path, allow_pickle=False) as z:
                keys = sorted(
                    [k for k in z.files if k.startswith("M") and k[1:].isdigit()],
                    key=lambda k: int(k[1:]),
                )
                if scalar(z["sample_id"]) != row.sample_id:
                    errors.append(f"id:{row.sample_id}")
                if scalar(z["sequence_sha256"]) != row.sequence_sha256:
                    errors.append(f"hash:{row.sample_id}")
                if int(np.asarray(z["sequence_length"]).reshape(-1)[0]) != int(row.length):
                    errors.append(f"length:{row.sample_id}")
                if len(keys) != len(args.motif_indices):
                    errors.append(f"matrix_count:{row.sample_id}:{len(keys)}")
                if any(
                    z[key].shape != (int(row.length), int(row.length))
                    for key in keys
                ):
                    errors.append(f"shape:{row.sample_id}")
                if not np.array_equal(
                    np.asarray(z["motif_indices"], dtype=np.int32),
                    args.motif_indices,
                ):
                    errors.append(f"motif_indices:{row.sample_id}")
                if scalar(z["selection_method"]) != FIXED_SELECTION_METHOD:
                    errors.append(f"selection_method:{row.sample_id}")
                if int(np.asarray(z["format_version"]).reshape(-1)[0]) < 3:
                    errors.append(f"format_version:{row.sample_id}")
        except Exception as exc:
            errors.append(
                f"load:{row.sample_id}:{type(exc).__name__}:{exc}"
            )

    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:30]))

    final = merged[
        [
            "sample_id",
            "sequence",
            "label",
            "group_id",
            "npz_path",
            "split",
            "length",
            "sequence_sha256",
        ]
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)

    print(final.groupby(["split", "label"]).size().to_string())
    print(f"Validated {len(final)} samples")
    print(f"Fixed motif indices {args.motif_indices.tolist()}")
    print(f"Selection method {FIXED_SELECTION_METHOD}")
    print(f"Compressed storage {total_bytes / 1024**3:.3f} GiB")
    print(output.resolve())


if __name__ == "__main__":
    main()
