from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANONICAL_AA = "ARNDCQEGHILKMFPSTWYV"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 21
VOCAB_SIZE = 22
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(CANONICAL_AA)}
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}
ID_TO_AA[PAD_TOKEN_ID] = "<PAD>"
ID_TO_AA[UNK_TOKEN_ID] = "<UNK>"
ALLOWED_AMBIGUOUS = set("XBZUOJ")


@dataclass(frozen=True)
class TokenizedSequence:
    tokens: np.ndarray
    true_length: int
    unknown_count: int


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def validate_sequence(sequence: str, sample_id: str = "") -> None:
    if not sequence:
        raise ValueError(f"Empty sequence for sample {sample_id!r}")
    invalid = sorted(set(sequence) - set(CANONICAL_AA) - ALLOWED_AMBIGUOUS)
    if invalid:
        raise ValueError(
            f"Invalid residue(s) {invalid} in sample {sample_id!r}. "
            "Remove gaps/stops or explicitly map them before training."
        )


def encode_sequence(sequence: str, max_length: int, sample_id: str = "") -> TokenizedSequence:
    sequence = normalize_sequence(sequence)
    validate_sequence(sequence, sample_id)
    if len(sequence) > max_length:
        raise ValueError(
            f"Sequence {sample_id!r} has length {len(sequence)}, above max_length={max_length}. "
            "This package intentionally refuses silent truncation. Increase --seq-len or prepare an explicit windowing dataset."
        )
    tokens = np.zeros(max_length, dtype=np.int64)
    unknown_count = 0
    for i, residue in enumerate(sequence):
        token = AA_TO_ID.get(residue, UNK_TOKEN_ID)
        unknown_count += int(token == UNK_TOKEN_ID)
        tokens[i] = token
    return TokenizedSequence(tokens=tokens, true_length=len(sequence), unknown_count=unknown_count)


def tokenizer_metadata() -> dict:
    return {
        "canonical_order": CANONICAL_AA,
        "pad_token_id": PAD_TOKEN_ID,
        "unknown_token_id": UNK_TOKEN_ID,
        "vocab_size": VOCAB_SIZE,
        "aa_to_id": AA_TO_ID,
    }
