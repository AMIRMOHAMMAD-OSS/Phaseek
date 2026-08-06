#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phaseek_v2.fegs_fast import (
    DEFAULT_FIXED_MOTIF_INDICES,
    FIXED_SELECTION_METHOD,
    FastFEGSExtractor,
)


def safe_filename(sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode()).hexdigest()[:12]
    return f"{slug[:90]}__{digest}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--m-mat", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    row = pd.read_csv(args.manifest).iloc[0]
    sample_id = str(row.sample_id)
    sequence = str(row.sequence)
    sequence_hash = str(row.sequence_sha256)

    extractor = FastFEGSExtractor(args.m_mat)
    result = extractor.extract_selected_graphs(
        sample_id,
        sequence,
        motif_indices=DEFAULT_FIXED_MOTIF_INDICES,
    )

    assert len(result.matrices) == 10
    assert np.array_equal(result.motif_indices, np.arange(10, dtype=np.int32))
    assert all(
        matrix.shape == (len(sequence), len(sequence))
        for matrix in result.matrices
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_filename(sample_id)

    payload = {
        f"M{i}": matrix.astype(np.float16, copy=False)
        for i, matrix in enumerate(result.matrices)
    }
    payload.update(
        sample_id=np.asarray(sample_id),
        sequence_length=np.asarray(len(sequence), dtype=np.int32),
        sequence_sha256=np.asarray(sequence_hash),
        motif_indices=result.motif_indices,
        motif_orderings=result.motif_orderings,
        shap_rank=np.arange(1, 11, dtype=np.int32),
        selection_method=np.asarray(FIXED_SELECTION_METHOD),
        storage_dtype=np.asarray("float16"),
        format_version=np.asarray(3, dtype=np.int32),
    )
    np.savez_compressed(output_path, **payload)

    print(
        {
            "sample_id": result.sample_id,
            "length": result.sequence_length,
            "motif_indices": result.motif_indices.tolist(),
            "motif_orderings": result.motif_orderings.tolist(),
            "selection_method": result.selection_method,
            "npz_path": str(output_path.resolve()),
            "npz_bytes": output_path.stat().st_size,
        }
    )


if __name__ == "__main__":
    main()
