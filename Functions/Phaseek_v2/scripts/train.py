#!/usr/bin/env python3
from __future__ import annotations

import argparse

from phaseek_v2.config import ModelConfig, TrainConfig, choose_model_profile
from phaseek_v2.data import read_manifest
from phaseek_v2.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the corrected Phaseek v2 graph-biased transformer")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--profile", choices=["auto", "small", "base"], default="auto")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")
    parser.add_argument("--graph-mixer", choices=["shared", "layerwise"], default="shared")
    parser.add_argument("--mixture-tau", type=float, default=1.0)
    parser.add_argument("--mixture-l2", type=float, default=1e-4)
    parser.add_argument("--mixture-init-std", type=float, default=0.0)
    parser.add_argument("--beta-init", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=0, help="0 selects from training-set size")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects from GPU memory")
    parser.add_argument("--effective-batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=3)
    parser.add_argument("--matrix-cache-items", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0, help="0 selects from effective batch size")
    parser.add_argument("--graph-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--balance", choices=["auto", "none", "weighted"], default="auto")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    parser.add_argument("--selection-metric", choices=["pr_auc", "roc_auc", "loss"], default="pr_auc")
    parser.add_argument("--matrix-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--allow-missing-matrices", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    n_train = int((manifest.split == "train").sum())
    profile = choose_model_profile(n_train, args.profile)
    n_layer = args.layers or profile["n_layer"]
    n_head = args.heads or profile["n_head"]
    n_embd = args.embedding_dim or profile["n_embd"]

    model_config = ModelConfig(
        block_size=args.seq_len,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        topk_m=args.topk,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        pooling=args.pooling,
        graph_mixer=args.graph_mixer,
        mixture_tau=args.mixture_tau,
        mixture_l2=args.mixture_l2,
        mixture_init_std=args.mixture_init_std,
        beta_init=args.beta_init,
    )
    train_config = TrainConfig(
        seed=args.seed,
        deterministic=args.deterministic,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch,
        matrix_cache_items=args.matrix_cache_items,
        learning_rate=args.lr,
        graph_lr_multiplier=args.graph_lr_multiplier,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        balance_mode=args.balance,
        amp=args.amp,
        selection_metric=args.selection_metric,
        strict_matrices=not args.allow_missing_matrices,
        matrix_dtype=args.matrix_dtype,
        compile_model=args.compile,
    )
    report = train_model(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_config=model_config,
        train_config=train_config,
        resume_path=args.resume,
    )
    print("Final validation metrics:")
    for key, value in report["validation"].items():
        print(f"  {key}: {value}")
    if report.get("test") is not None:
        print("Final test metrics:")
        for key, value in report["test"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
