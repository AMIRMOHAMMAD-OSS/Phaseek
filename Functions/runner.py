from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
V1_DIR = ROOT / "Phaseek_v1"
V2_DIR = ROOT / "Phaseek_v2"
RESULTS_DIR = ROOT / "Results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseek_v2.config import ModelConfig
from phaseek_v2.fegs_fast import FastFEGSExtractor
from phaseek_v2.matrices import standardize_valid_matrix
from phaseek_v2.model import PhaseekV2Classifier
from phaseek_v2.tokenizer import encode_sequence

warnings.filterwarnings("ignore")

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

_V1_CLF = None
_V1_MODEL = None
_V2_MODEL = None
_V2_CONFIG = None
_V2_CHECKPOINT = None
_V2_EXTRACTOR = None
_V2_DEVICE = None



def clean_sequence(sequence: str) -> str:
    sequence = "".join(ch.upper() for ch in sequence if ch.upper() in ALPHABET)
    if not sequence:
        raise ValueError("The sequence does not contain valid amino-acid residues.")
    return sequence


def load_v1():
    global _V1_CLF, _V1_MODEL

    if _V1_CLF is not None and _V1_MODEL is not None:
        return _V1_CLF, _V1_MODEL

    import random

    root_path = str(ROOT.resolve())
    v1_path = V1_DIR.resolve()

    # Never import the duplicated modules from Functions/Phaseek_v1.
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != v1_path
    ]

    # Force the original v1 modules in Functions/ to be imported first.
    if root_path in sys.path:
        sys.path.remove(root_path)
    sys.path.insert(0, root_path)

    # Remove a wrong cached import when this module is used interactively.
    for module_name in ("classifier_fgs", "XGBoost", "Configue"):
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)

        if module_file and Path(module_file).resolve().parent != ROOT.resolve():
            del sys.modules[module_name]

    import classifier_fgs
    import XGBoost as xgboost_module

    classifier_path = Path(classifier_fgs.__file__).resolve()
    xgboost_path = Path(xgboost_module.__file__).resolve()

    if classifier_path.parent != ROOT.resolve():
        raise ImportError(
            f"Wrong classifier_fgs imported: {classifier_path}"
        )

    if xgboost_path.parent != ROOT.resolve():
        raise ImportError(
            f"Wrong XGBoost imported: {xgboost_path}"
        )

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _V1_CLF = xgboost_module.XGBoost.XGM()
    _V1_MODEL = classifier_fgs.transformer("c")

    return _V1_CLF, _V1_MODEL


def enhanced_mean(scores):
    scores = np.asarray(scores, dtype=float)
    weights = 1.0 / (1.0 + np.exp(-50.0 * (2.0 * scores - 1.0)))
    denominator = float(weights.sum())
    if denominator == 0:
        return float(scores.mean())
    return float(np.sum(scores * weights) / denominator)


def pad_v1(values):
    values = list(values)
    return values + [0.0] * (5537 - len(values))


def combine_v1_score(residue_scores, sequence_score, clf):
    def scorer(values):
        vector = np.array(
            pad_v1(values) + [sequence_score],
            dtype=float,
        ).reshape(1, 5538)

        probability = float(clf.predict_proba(vector)[:, 1][0])
        nonzero = [value for value in vector.ravel() if value != 0.0]
        local_score = enhanced_mean(nonzero[:-1]) if len(nonzero) > 1 else 0.0

        return (
            0.3 * float(sequence_score)
            + 0.4 * local_score
            + 0.3 * probability
        )

    if len(residue_scores) > 5537:
        chunks = [
            residue_scores[start:start + 5537]
            for start in range(0, len(residue_scores), 5537)
        ]
        return enhanced_mean([scorer(chunk) for chunk in chunks])

    return float(scorer(residue_scores))


def score1(index, scores, half_window, length):
    i = index - 1

    if i <= half_window:
        return sum(scores[:i + 1]) / (i + 1)

    if half_window + 1 <= i < length - half_window:
        return (
            sum(scores[i - half_window:i + 1])
            / (half_window + 1)
        )

    return (
        sum(scores[i - half_window:length - half_window + 1])
        / (length - i + 1)
    )


def adjust_residue_score(value, sequence_score):
    if sequence_score > 0.7:
        return value * np.exp(-1.2 * (value - sequence_score))
    return value


def score_v1_sequence(sequence: str):
    clf, model = load_v1()

    length = len(sequence)
    if length < 3:
        raise ValueError("Phaseek v1 requires at least 3 residues.")

    window = max(5, min(50, int(np.ceil(0.1 * length))))
    half_window = window // 2

    sequence_chunks = [
        sequence[i:i + half_window]
        for i in range(length - half_window + 1)
    ]

    if len(sequence_chunks) > 700:
        predictions = [
            model.predict_proba(sequence_chunks[i:i + 700])
            for i in range(0, len(sequence_chunks), 700)
        ]
        local_scores = np.concatenate(
            [np.asarray(item, dtype=float).reshape(-1) for item in predictions]
        )
    else:
        local_scores = np.asarray(
            model.predict_proba(sequence_chunks),
            dtype=float,
        ).reshape(-1)

    raw_sequence_score = np.asarray(
        model.predict_proba([sequence]),
        dtype=float,
    ).reshape(-1)

    sequence_score = float(raw_sequence_score[0])

    residue_scores = [
        adjust_residue_score(
            score1(position, local_scores, half_window, length),
            sequence_score,
        )
        for position in range(1, length + 1)
    ]

    residue_scores = [
        float(np.asarray(value).reshape(-1)[0])
        for value in residue_scores
    ]

    final_score = combine_v1_score(
        residue_scores,
        sequence_score,
        clf,
    )

    return float(final_score), residue_scores


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device_name)


def load_v2(checkpoint_path: Path, m_mat_path: Path, device_name: str):
    global _V2_MODEL
    global _V2_CONFIG
    global _V2_CHECKPOINT
    global _V2_EXTRACTOR
    global _V2_DEVICE

    if _V2_MODEL is not None:
        return (
            _V2_MODEL,
            _V2_CONFIG,
            _V2_CHECKPOINT,
            _V2_EXTRACTOR,
            _V2_DEVICE,
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Phaseek v2 checkpoint not found: {checkpoint_path}"
        )

    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = ModelConfig(**checkpoint["model_config"])
    model = PhaseekV2Classifier(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    extractor = FastFEGSExtractor(m_mat_path)

    _V2_MODEL = model
    _V2_CONFIG = config
    _V2_CHECKPOINT = checkpoint
    _V2_EXTRACTOR = extractor
    _V2_DEVICE = device

    return model, config, checkpoint, extractor, device


def make_windows(sequence: str, window_size: int, stride: int):
    if len(sequence) <= window_size:
        return [(0, sequence)]

    starts = list(range(0, len(sequence) - window_size + 1, stride))
    final_start = len(sequence) - window_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return [
        (start, sequence[start:start + window_size])
        for start in starts
    ]


@torch.inference_mode()
def score_v2_sequence(
    sequence: str,
    checkpoint_path: Path,
    m_mat_path: Path,
    device_name: str = "auto",
    window_size: int = 512,
    stride: int = 256,
    aggregation: str = "mean",
):
    model, config, checkpoint, extractor, device = load_v2(
        checkpoint_path,
        m_mat_path,
        device_name,
    )

    window_size = min(int(window_size), int(config.block_size))
    stride = int(stride)

    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if stride < 1 or stride > window_size:
        raise ValueError("stride must be between 1 and window_size.")

    windows = make_windows(sequence, window_size, stride)
    window_results = []

    for index, (start, window_sequence) in enumerate(windows, start=1):
        sample_id = f"window_{index}"

        tokenized = encode_sequence(
            window_sequence,
            max_length=config.block_size,
            sample_id=sample_id,
        )

        extracted = extractor.extract_selected_graphs(
            sample_id=sample_id,
            sequence=window_sequence,
            motif_indices=np.arange(config.topk_m, dtype=np.int32),
        )

        matrices = np.stack(
            [
                standardize_valid_matrix(
                    matrix,
                    valid_length=len(window_sequence),
                    target_length=config.block_size,
                )
                for matrix in extracted.matrices
            ],
            axis=0,
        ).astype(np.float32)

        tokens_tensor = torch.from_numpy(
            tokenized.tokens
        ).unsqueeze(0).to(device)

        matrices_tensor = torch.from_numpy(
            matrices
        ).unsqueeze(0).to(device)

        logits, _ = model(tokens_tensor, matrices_tensor)
        probability = float(
            torch.softmax(logits.float(), dim=-1)[0, 1].item()
        )

        window_results.append(
            {
                "window": index,
                "start": start + 1,
                "end": start + len(window_sequence),
                "score": probability,
            }
        )

    probabilities = np.asarray(
        [item["score"] for item in window_results],
        dtype=float,
    )

    if aggregation == "max":
        final_score = float(probabilities.max())
    else:
        final_score = float(probabilities.mean())

    threshold = float(checkpoint.get("validation_threshold", 0.5))

    return final_score, threshold, window_results


def result_directory(directory: str, sample_id: str):
    safe_directory = str(directory).replace("..", "_")
    safe_id = str(sample_id).replace("/", "_").replace("\\", "_")

    path = RESULTS_DIR / safe_directory / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def score_one(
    sequence: str,
    sample_id: str,
    model_version: str,
    directory: str,
    checkpoint_path: Path,
    m_mat_path: Path,
    device_name: str,
    window_size: int,
    stride: int,
    aggregation: str,
):
    sequence = clean_sequence(sequence)

    v1_score, residue_scores = score_v1_sequence(sequence)

    if model_version == "v2":
        global_score, threshold, window_results = score_v2_sequence(
            sequence,
            checkpoint_path=checkpoint_path,
            m_mat_path=m_mat_path,
            device_name=device_name,
            window_size=window_size,
            stride=stride,
            aggregation=aggregation,
        )
    else:
        global_score = v1_score
        threshold = 0.5
        window_results = []

    output_dir = result_directory(directory, sample_id)

    pd.DataFrame(
        {
            "scores": residue_scores,
            "seq": list(sequence),
        }
    ).to_csv(output_dir / "scores.csv", index=False)

    if window_results:
        pd.DataFrame(window_results).to_csv(
            output_dir / "window_scores.csv",
            index=False,
        )

    summary = {
        "id": sample_id,
        "sequence_length": len(sequence),
        "model": model_version,
        "score": float(global_score),
        "threshold": float(threshold),
        "prediction": int(global_score >= threshold),
        "residue_level_model": "v1",
        "v1_score": float(v1_score),
        "window_aggregation": (
            aggregation if model_version == "v2" else None
        ),
        "number_of_windows": len(window_results),
    }

    with (output_dir / "prediction.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary, residue_scores, window_results


def read_fasta(path: Path, limit: int):
    records = []

    for record in SeqIO.parse(str(path), "fasta"):
        records.append((str(record.id), str(record.seq)))
        if len(records) >= limit:
            break

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Phaseek v1/v2 LLPS scoring"
    )

    parser.add_argument(
        "--model",
        choices=["v1", "v2"],
        default="v1",
    )
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--fasta", type=str, default="")
    parser.add_argument("--id", type=str, default="DefaultID")
    parser.add_argument("--directory", type=str, default="DefaultDir")
    parser.add_argument(
        "--end_sequence",
        "--end-sequence",
        dest="end_sequence",
        type=int,
        default=500,
    )
    parser.add_argument("--plot", default="True")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=V2_DIR / "model" / "best.pt",
    )
    parser.add_argument(
        "--m-mat",
        type=Path,
        default=ROOT / "M.mat",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max"],
        default="mean",
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta) if args.fasta else None

    if fasta_path is not None:
        rows = []
        records = read_fasta(fasta_path, args.end_sequence)

        for sample_id, sequence in tqdm(records):
            try:
                summary, residue_scores, window_results = score_one(
                    sequence=sequence,
                    sample_id=sample_id,
                    model_version=args.model,
                    directory=args.directory,
                    checkpoint_path=args.checkpoint,
                    m_mat_path=args.m_mat,
                    device_name=args.device,
                    window_size=args.window_size,
                    stride=args.stride,
                    aggregation=args.aggregation,
                )

                rows.append(
                    {
                        "id": sample_id,
                        "seq": clean_sequence(sequence),
                        "model": args.model,
                        "LLPS_score": summary["score"],
                        "prediction": summary["prediction"],
                        "threshold": summary["threshold"],
                        "Residue-level score": json.dumps(
                            residue_scores
                        ),
                        "window_scores": json.dumps(
                            window_results
                        ),
                    }
                )

            except Exception as error:
                rows.append(
                    {
                        "id": sample_id,
                        "seq": clean_sequence(sequence),
                        "model": args.model,
                        "LLPS_score": None,
                        "prediction": None,
                        "threshold": None,
                        "Residue-level score": None,
                        "window_scores": None,
                        "error": str(error),
                    }
                )

        output_dir = result_directory(
            args.directory,
            args.id or "FASTA_results",
        )

        output_path = output_dir / "LLPS_prediction_of_seqs.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)

        print("Analysis complete.")
        print(f"Results: {output_path}")
        return

    if not args.sequence:
        raise ValueError(
            "Provide either --sequence or --fasta."
        )

    summary, _, _ = score_one(
        sequence=args.sequence,
        sample_id=args.id,
        model_version=args.model,
        directory=args.directory,
        checkpoint_path=args.checkpoint,
        m_mat_path=args.m_mat,
        device_name=args.device,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
    )

    print(f"Score: {summary['score']}")


if __name__ == "__main__":
    main()
