#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path: Path) -> dict:
    metrics_path = path / "final_metrics.json"
    history_path = path / "history.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    report = json.loads(metrics_path.read_text())
    validation = report["validation"]
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    result = {
        "run": path.name,
        "path": str(path),
        "pr_auc": validation.get("pr_auc"),
        "roc_auc": validation.get("roc_auc"),
        "mcc": validation.get("mcc"),
        "f1": validation.get("f1"),
        "precision": validation.get("precision"),
        "recall": validation.get("recall"),
        "specificity": validation.get("specificity"),
        "threshold": validation.get("threshold"),
        "epochs_ran": int(len(history)),
        "best_epoch_pr_auc": None,
        "mixture_max_deviation": None,
        "mixture_mean_entropy": None,
        "beta_mean_abs": None,
        "beta_max_abs": None,
    }
    if not history.empty and "val_pr_auc" in history:
        result["best_epoch_pr_auc"] = int(history.loc[history.val_pr_auc.idxmax(), "epoch"])
    mixture_path = path / "mixture_weights.npy"
    if mixture_path.exists():
        weights = np.load(mixture_path).astype(np.float64)
        uniform = 1.0 / weights.shape[-1]
        result["mixture_max_deviation"] = float(np.max(np.abs(weights - uniform)))
        entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=-1)
        result["mixture_mean_entropy"] = float(np.mean(entropy))
    beta_path = path / "beta_by_layer_head.npy"
    if beta_path.exists():
        beta = np.load(beta_path).astype(np.float64)
        result["beta_mean_abs"] = float(np.mean(np.abs(beta)))
        result["beta_max_abs"] = float(np.max(np.abs(beta)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed Phaseek experiment directories")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    frame = pd.DataFrame([load_run(path.resolve()) for path in args.runs])
    order = [
        "run", "pr_auc", "roc_auc", "mcc", "f1", "precision", "recall",
        "specificity", "threshold", "epochs_ran", "best_epoch_pr_auc",
        "mixture_max_deviation", "mixture_mean_entropy", "beta_mean_abs",
        "beta_max_abs", "path",
    ]
    frame = frame[order].sort_values("pr_auc", ascending=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
