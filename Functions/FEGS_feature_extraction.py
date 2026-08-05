from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform

from .tokenizer import CANONICAL_AA, normalize_sequence, validate_sequence


DEFAULT_FIXED_MOTIF_INDICES = np.arange(10, dtype=np.int32)
FIXED_SELECTION_METHOD = "fixed_first10_shap_ordered_Mmat"


@dataclass
class GraphExtractionResult:
    sample_id: str
    sequence_length: int
    matrices: list[np.ndarray]
    motif_indices: np.ndarray
    motif_orderings: np.ndarray
    selection_method: str


class FastFEGSExtractor:
    def __init__(self, m_mat_path: str | Path, legacy_wrap_dpc: bool = False):
        mat = loadmat(str(m_mat_path))
        if "M" not in mat:
            raise KeyError(f"M.mat file {m_mat_path} does not contain variable 'M'")

        raw_motifs = mat["M"].flatten()
        motifs: list[str] = []
        for raw in raw_motifs:
            value = raw.item() if hasattr(raw, "item") else raw
            motif = str(value)
            if len(motif) != 20 or set(motif) != set(CANONICAL_AA):
                raise ValueError(
                    "Every motif in M.mat must be a permutation of the 20 canonical "
                    f"amino acids; received {motif!r}"
                )
            motifs.append(motif)

        if len(motifs) < 10:
            raise ValueError(
                f"M.mat contains only {len(motifs)} motifs; at least 10 are required"
            )

        self.motifs = motifs
        self.num_motifs = len(motifs)
        self.legacy_wrap_dpc = legacy_wrap_dpc
        self.coordinates = self._coordinates()

        code_of = {aa: i for i, aa in enumerate(CANONICAL_AA)}
        self.code_of = code_of
        self.rank = np.zeros((self.num_motifs, 20), dtype=np.intp)
        for motif_index, motif in enumerate(motifs):
            for position, amino_acid in enumerate(motif):
                self.rank[motif_index, code_of[amino_acid]] = position

    @staticmethod
    def _coordinates() -> np.ndarray:
        angles = np.arange(20, dtype=np.float64) * (2.0 * np.pi / 20.0)
        return np.column_stack([np.cos(angles), np.sin(angles), np.ones(20)])

    def encode(self, sequence: str, sample_id: str = "") -> np.ndarray:
        sequence = normalize_sequence(sequence)
        validate_sequence(sequence, sample_id)
        return np.asarray(
            [self.code_of.get(residue, -1) for residue in sequence],
            dtype=np.intp,
        )

    def _graphical_curves_from_rank(
        self,
        sequence_codes: np.ndarray,
        motif_rank: np.ndarray,
    ) -> np.ndarray:
        """Create curves only for the supplied motif-rank rows."""
        length = int(sequence_codes.shape[0])
        known = sequence_codes >= 0
        safe = np.where(known, sequence_codes, 0)

        motif_positions = motif_rank[:, safe]
        contributions = self.coordinates[motif_positions].copy()

        if not known.all():
            indices = np.arange(length)
            mid_unknown = (~known) & (indices >= 1)
            first_unknown = (~known) & (indices == 0)
            if mid_unknown.any():
                contributions[:, mid_unknown, :] = np.array([0.0, 0.0, 1.0])
            if first_unknown.any():
                contributions[:, first_unknown, :] = 0.0

        if length > 1:
            previous = motif_positions[:, :-1]
            current = motif_positions[:, 1:]
            transition = self.coordinates[previous] + (
                self.coordinates[current] - self.coordinates[previous]
            ) / 20.0
            both_known = known[:-1] & known[1:]
            transition *= both_known[None, :, None]
            running_mean = np.cumsum(transition, axis=1) / np.arange(
                1, length
            )[None, :, None]
            contributions[:, 1:, :] += running_mean

        return np.cumsum(contributions, axis=1)

    def graphical_curves(self, sequence_codes: np.ndarray) -> np.ndarray:
        """Classical FEGS path: curves for all motifs."""
        return self._graphical_curves_from_rank(sequence_codes, self.rank)

    def graphical_curves_selected(
        self,
        sequence_codes: np.ndarray,
        motif_indices: Sequence[int] = DEFAULT_FIXED_MOTIF_INDICES,
    ) -> np.ndarray:
        indices = self._validate_motif_indices(motif_indices)
        return self._graphical_curves_from_rank(sequence_codes, self.rank[indices])

    def _validate_motif_indices(
        self,
        motif_indices: Sequence[int],
    ) -> np.ndarray:
        indices = np.asarray(motif_indices, dtype=np.int32).reshape(-1)
        if len(indices) == 0:
            raise ValueError("At least one motif index is required")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Motif indices must be unique, received {indices.tolist()}")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_motifs:
            raise ValueError(
                f"Motif indices {indices.tolist()} exceed M.mat range "
                f"0..{self.num_motifs - 1}"
            )
        return indices

    @staticmethod
    def graph_matrix(curve: np.ndarray) -> np.ndarray:
        length = curve.shape[0]
        if length == 0:
            raise ValueError("Cannot create a graph matrix for an empty sequence")
        if length == 1:
            return np.zeros((1, 1), dtype=np.float32)

        euclidean = squareform(pdist(curve))
        step_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(step_lengths)))
        arc_distance = np.abs(cumulative[:, None] - cumulative[None, :])
        denominator = arc_distance + np.eye(length, dtype=np.float64)
        matrix = np.divide(
            euclidean,
            denominator,
            out=np.zeros_like(euclidean),
            where=denominator != 0,
        )
        matrix = 0.5 * (matrix + matrix.T)
        return matrix.astype(np.float32)

    @staticmethod
    def dominant_score(matrix: np.ndarray) -> float:
        length = matrix.shape[0]
        if length <= 1:
            return 0.0
        matrix64 = np.asarray(matrix, dtype=np.float64)
        try:
            if length < 32:
                value = eigvalsh(
                    matrix64,
                    subset_by_index=[length - 1, length - 1],
                )[0]
            else:
                value = eigsh(
                    matrix64,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                    tol=1e-6,
                    maxiter=max(1000, length * 20),
                )[0]
        except Exception:
            value = eigvalsh(
                matrix64,
                subset_by_index=[length - 1, length - 1],
            )[0]
        return float(value) / length

    def extract_selected_graphs(
        self,
        sample_id: str,
        sequence: str,
        motif_indices: Sequence[int] = DEFAULT_FIXED_MOTIF_INDICES,
    ) -> GraphExtractionResult:
        indices = self._validate_motif_indices(motif_indices)
        normalized_sequence = normalize_sequence(sequence)
        codes = self.encode(normalized_sequence, sample_id)
        curves = self._graphical_curves_from_rank(codes, self.rank[indices])
        matrices = [self.graph_matrix(curve) for curve in curves]
        orderings = np.asarray([self.motifs[i] for i in indices], dtype="<U20")

        return GraphExtractionResult(
            sample_id=sample_id,
            sequence_length=len(normalized_sequence),
            matrices=matrices,
            motif_indices=indices.copy(),
            motif_orderings=orderings,
            selection_method=FIXED_SELECTION_METHOD,
        )

    def extract_topk_graphs(
        self,
        sample_id: str,
        sequence: str,
        topk: int = 10,
    ) -> GraphExtractionResult:

        warnings.warn(
            "extract_topk_graphs now means fixed first-k SHAP-ordered M.mat rows; "
            "use extract_selected_graphs for explicit behavior",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.extract_selected_graphs(
            sample_id,
            sequence,
            motif_indices=np.arange(topk, dtype=np.int32),
        )

    def composition_features(
        self,
        sequence_codes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        length = len(sequence_codes)
        valid = sequence_codes[sequence_codes >= 0]
        aac = np.bincount(valid, minlength=20).astype(np.float64) / max(length, 1)
        dpc = np.zeros((20, 20), dtype=np.float64)
        if length > 1:
            if self.legacy_wrap_dpc:
                next_codes = np.roll(sequence_codes, -1)
                valid_pairs = (sequence_codes >= 0) & (next_codes >= 0)
                np.add.at(
                    dpc,
                    (sequence_codes[valid_pairs], next_codes[valid_pairs]),
                    1.0,
                )
            else:
                left = sequence_codes[:-1]
                right = sequence_codes[1:]
                valid_pairs = (left >= 0) & (right >= 0)
                np.add.at(
                    dpc,
                    (left[valid_pairs], right[valid_pairs]),
                    1.0,
                )
            dpc /= length - 1
        return aac, dpc

    def extract_features(self, sequences: Iterable[str]) -> np.ndarray:
        """Classical FEGS 158 eigenvalue + AAC + DPC feature vectors."""
        rows = []
        for sequence in sequences:
            codes = self.encode(sequence)
            curves = self.graphical_curves(codes)
            eigenvalues = np.asarray(
                [
                    self.dominant_score(self.graph_matrix(curve))
                    for curve in curves
                ],
                dtype=np.float64,
            )
            aac, dpc = self.composition_features(codes)
            rows.append(np.concatenate([eigenvalues, aac, dpc.ravel()]))
        return np.vstack(rows)
