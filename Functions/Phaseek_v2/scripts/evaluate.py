#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phaseek_v3.config import TrainConfig
from phaseek_v3.data import make_loader, read_manifest
from phaseek_v3.trainer import evaluate_loader, load_model_from_checkpoint
from phaseek_v3.utils import resolve_amp_dtype, setup_torch, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phaseek v3 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    args = parser.parse_args()

    device = setup_torch()
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    frame = read_manifest(args.manifest)
    frame = frame[frame.split == args.split].copy()
    train_config = TrainConfig(**checkpoint["train_config"])
    class_weight_values = checkpoint.get("class_weights")
    class_weights = (
        None
        if class_weight_values is None
        else torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )
    loader = make_loader(
        frame,
        max_length=model.config.block_size,
        topk_m=model.config.topk_m,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=train_config.prefetch_factor,
        cache_items=train_config.matrix_cache_items,
        matrix_dtype=train_config.matrix_dtype,
        strict_matrices=train_config.strict_matrices,
        shuffle=False,
        pin_memory=device.type == "cuda",
        seed=train_config.seed,
    )
    amp_enabled, amp_dtype, _ = resolve_amp_dtype(device, args.amp)
    threshold = float(checkpoint["validation_threshold"])
    metrics, predictions = evaluate_loader(
        model,
        loader,
        device,
        class_weights,
        train_config.label_smoothing,
        amp_enabled,
        amp_dtype,
        threshold=threshold,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"{args.split}_predictions.csv", index=False)
    write_json(output_dir / f"{args.split}_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
