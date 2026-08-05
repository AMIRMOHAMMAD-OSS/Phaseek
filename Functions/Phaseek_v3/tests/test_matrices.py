from pathlib import Path

import numpy as np
import pytest

from phaseek_v3.matrices import MatrixLoadConfig, MatrixStore, standardize_valid_matrix


def test_standardization_uses_only_valid_region():
    matrix = np.full((5, 5), 1000.0, dtype=np.float32)
    matrix[:3, :3] = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = standardize_valid_matrix(matrix, valid_length=3, target_length=5)
    assert abs(float(result[:3, :3].mean())) < 1e-6
    assert abs(float(result[:3, :3].std()) - 1.0) < 1e-6
    assert np.all(result[3:, :] == 0)
    assert np.all(result[:, 3:] == 0)


def test_npz_identity_is_checked(tmp_path: Path):
    path = tmp_path / "graph.npz"
    np.savez_compressed(
        path,
        M0=np.eye(3, dtype=np.float32),
        sample_id=np.asarray("actual"),
        sequence_length=np.asarray(3),
    )
    store = MatrixStore(MatrixLoadConfig(target_length=4, topk_m=1, strict=True))
    with pytest.raises(ValueError, match="ID mismatch"):
        store.load(path, sample_id="different", sequence_length=3)
