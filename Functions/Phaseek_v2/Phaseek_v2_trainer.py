from __future__ import annotations

import contextlib
import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm

from .config import (
    ModelConfig,
    TrainConfig,
    class_weights_from_counts,
    resolve_training_defaults,
)
from .data import make_loader, read_manifest
from .metrics import compute_metrics, select_threshold
from .model import PhaseekV3Classifier
from .tokenizer import tokenizer_metadata
from .utils import (
    atomic_torch_save,
    file_sha256,
    gpu_memory_gb,
    resolve_amp_dtype,
    set_seed,
    setup_torch,
    write_json,
)


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = min(total_steps - 1, max(1, int(total_steps * warmup_fraction)))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


@torch.inference_mode()
def evaluate_loader(
    model: PhaseekV3Classifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    threshold: float | None,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        matrices = batch["matrices"].to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits, auxiliary = model(tokens, matrices)
            loss = classification_loss(logits, labels, class_weights, label_smoothing)
            loss = loss + auxiliary["mixture_regularization"]
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
        total_loss += float(loss.item()) * len(labels)
        total_samples += len(labels)
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_ids.extend(batch["sample_ids"])

    labels_np = np.concatenate(all_labels)
    probabilities_np = np.concatenate(all_probabilities)
    average_loss = total_loss / max(1, total_samples)
    if threshold is None:
        threshold = select_threshold(labels_np, probabilities_np)
    result = compute_metrics(labels_np, probabilities_np, average_loss, threshold)
    predictions = pd.DataFrame(
        {
            "sample_id": all_ids,
            "label": labels_np,
            "probability_positive": probabilities_np,
            "prediction": (probabilities_np >= threshold).astype(int),
            "threshold": threshold,
        }
    )
    return result.to_dict(), predictions


def gradient_flow_smoke_test(
    model: PhaseekV3Classifier,
    batch: dict[str, Any],
    device: torch.device,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: GradScaler,
) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    matrices = batch["matrices"].to(device, non_blocking=True)
    with _autocast_context(device, amp_enabled, amp_dtype):
        logits, auxiliary = model(tokens, matrices)
        loss = classification_loss(logits, labels, class_weights, label_smoothing)
        loss = loss + auxiliary["mixture_regularization"]
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        # No optimizer is needed: divide the observed gradients by the current scale for reporting.
        scale = float(scaler.get_scale())
    else:
        scale = 1.0

    def grad_stats(parameter: torch.nn.Parameter) -> dict:
        grad = parameter.grad
        return {
            "present": grad is not None,
            "finite": bool(grad is not None and torch.isfinite(grad).all().item()),
            "norm": None if grad is None else float(grad.float().norm().item() / scale),
        }

    mixer_parameters = {
        name: grad_stats(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("mixer.")
    }
    report = {
        "loss": float(loss.detach().item()),
        "graph_mixer": model.config.graph_mixer,
        "mixer_parameters": mixer_parameters,
        "beta_by_block": [grad_stats(block.attn.beta) for block in model.blocks],
    }
    required = [*mixer_parameters.values(), *report["beta_by_block"]]
    if not required or not all(item["present"] and item["finite"] for item in required):
        raise RuntimeError(f"Graph-path gradient flow test failed: {report}")
    if any(item["norm"] == 0.0 for item in mixer_parameters.values()):
        raise RuntimeError(
            "At least one graph-mixture gradient is exactly zero on the real smoke-test batch. "
            "Inspect graph diversity and the attention bias path before training."
        )
    # Preserve convenient legacy fields for the validated shared mixer.
    if "mixer.alpha" in mixer_parameters:
        report["mixer_alpha"] = mixer_parameters["mixer.alpha"]
    if "mixer.delta" in mixer_parameters:
        report["mixer_delta"] = mixer_parameters["mixer.delta"]
    model.zero_grad(set_to_none=True)
    return report


def _selection_value(metrics: dict, selection_metric: str) -> float:
    if selection_metric == "loss":
        return -float(metrics["loss"])
    return float(metrics[selection_metric])


def _checkpoint_payload(
    model: PhaseekV3Classifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_score: float,
    patience_count: int,
    threshold: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    class_weights: tuple[float, float] | None,
    manifest_path: Path,
    history: list[dict],
) -> dict:
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "patience_count": int(patience_count),
        "validation_threshold": float(threshold),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "class_weights": class_weights,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "tokenizer": tokenizer_metadata(),
        "history": history,
    }


def train_model(
    manifest_path: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    resume_path: str | Path | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_config.seed, train_config.deterministic)
    device = setup_torch()

    manifest = read_manifest(manifest_path)
    splits = {name: manifest[manifest.split == name].copy() for name in ("train", "val", "test")}
    for name in ("train", "val"):
        frame = splits[name]
        if frame.empty:
            raise ValueError(f"Manifest split {name!r} is empty")
        if set(frame.label.unique()) != {0, 1}:
            raise ValueError(f"Manifest split {name!r} must contain both classes")
    if not splits["test"].empty and set(splits["test"].label.unique()) != {0, 1}:
        raise ValueError("Optional test split must contain both classes")

    train_config = resolve_training_defaults(
        train_config,
        n_train=len(splits["train"]),
        gpu_memory_gb=gpu_memory_gb(device),
    )
    model_config.validate()
    train_config.validate()

    n_negative = int((splits["train"].label == 0).sum())
    n_positive = int((splits["train"].label == 1).sum())
    class_weight_values = class_weights_from_counts(
        n_negative,
        n_positive,
        mode=train_config.balance_mode,
    )
    class_weights = (
        None
        if class_weight_values is None
        else torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )

    accumulation_steps = math.ceil(train_config.effective_batch_size / train_config.batch_size)
    actual_effective_batch = train_config.batch_size * accumulation_steps
    pin_memory = device.type == "cuda"
    loader_kwargs = dict(
        max_length=model_config.block_size,
        topk_m=model_config.topk_m,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        cache_items=train_config.matrix_cache_items,
        matrix_dtype=train_config.matrix_dtype,
        strict_matrices=train_config.strict_matrices,
        pin_memory=pin_memory,
        seed=train_config.seed,
    )
    train_loader = make_loader(splits["train"], shuffle=True, **loader_kwargs)
    val_loader = make_loader(splits["val"], shuffle=False, **loader_kwargs)
    test_loader = (
        None
        if splits["test"].empty
        else make_loader(splits["test"], shuffle=False, **loader_kwargs)
    )

    model = PhaseekV3Classifier(model_config).to(device)
    if train_config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
    raw_model: PhaseekV3Classifier = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[attr-defined]

    optimizer_kwargs: dict[str, Any] = {
        "params": raw_model.optimizer_groups(
            train_config.weight_decay,
            base_lr=train_config.learning_rate,
            graph_lr_multiplier=train_config.graph_lr_multiplier,
        ),
        "lr": train_config.learning_rate,
        "betas": (0.9, 0.95),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(**optimizer_kwargs)

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * train_config.epochs)
    scheduler = cosine_schedule(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_fraction=train_config.warmup_fraction,
        min_lr_ratio=train_config.min_learning_rate_ratio,
    )
    amp_enabled, amp_dtype, use_scaler = resolve_amp_dtype(device, train_config.amp)
    scaler = GradScaler("cuda", enabled=use_scaler)

    resolved = {
        "device": str(device),
        "gpu_memory_gb": gpu_memory_gb(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": None if amp_dtype is None else str(amp_dtype),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "class_counts_train": {"negative": n_negative, "positive": n_positive},
        "class_weights": class_weight_values,
        "accumulation_steps": accumulation_steps,
        "actual_effective_batch_size": actual_effective_batch,
        "parameter_count": sum(p.numel() for p in raw_model.parameters()),
        "optimizer_groups": [
            {
                "group_name": group.get("group_name", f"group_{index}"),
                "initial_lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(sum(p.numel() for p in group["params"])),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
    }
    write_json(output_dir / "resolved_config.json", resolved)

    start_epoch = 0
    best_score = -float("inf")
    patience_count = 0
    best_threshold = 0.5
    history: list[dict] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        patience_count = int(checkpoint.get("patience_count", 0))
        best_threshold = float(checkpoint.get("validation_threshold", 0.5))
        history = list(checkpoint.get("history", []))

    smoke_batch = next(iter(train_loader))
    gradient_report = gradient_flow_smoke_test(
        raw_model,
        smoke_batch,
        device,
        class_weights,
        train_config.label_smoothing,
        amp_enabled,
        amp_dtype,
        scaler,
    )
    write_json(output_dir / "gradient_flow_report.json", gradient_report)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.csv"

    previous_freeze_state: bool | None = None
    for epoch in range(start_epoch, train_config.epochs):
        freeze_backbone = epoch < train_config.freeze_backbone_epochs
        raw_model.set_backbone_frozen(freeze_backbone)
        if freeze_backbone != previous_freeze_state:
            stage = "graph/head warm-up (backbone frozen)" if freeze_backbone else "full fine-tuning"
            trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in raw_model.parameters())
            print(f"Training stage: {stage}; trainable parameters {trainable}/{total}")
            if previous_freeze_state is True and not freeze_backbone:
                patience_count = 0
                print("Backbone unfrozen; early-stopping patience reset.")
            previous_freeze_state = freeze_backbone
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_samples = 0
        accumulation_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{train_config.epochs}", ncols=110)

        for batch_index, batch in enumerate(progress):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            matrices = batch["matrices"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled, amp_dtype):
                logits, auxiliary = model(tokens, matrices)
                full_loss = classification_loss(
                    logits,
                    labels,
                    class_weights,
                    train_config.label_smoothing,
                ) + auxiliary["mixture_regularization"]
                scaled_for_accumulation = full_loss / accumulation_steps

            scaler.scale(scaled_for_accumulation).backward()
            accumulation_count += 1
            running_loss += float(full_loss.detach().item()) * len(labels)
            running_samples += len(labels)
            should_step = accumulation_count == accumulation_steps or batch_index == len(train_loader) - 1

            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if accumulation_count < accumulation_steps:
                    correction = accumulation_steps / accumulation_count
                    for parameter in raw_model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    train_config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulation_count = 0
            else:
                grad_norm = torch.tensor(float("nan"))

            current_lrs = scheduler.get_last_lr()
            progress.set_postfix(
                loss=f"{running_loss / max(1, running_samples):.4f}",
                lr=f"{current_lrs[0]:.2e}",
                glr=f"{current_lrs[-1]:.2e}",
                grad=f"{float(grad_norm):.2f}" if torch.isfinite(grad_norm) else "-",
            )

        val_metrics, val_predictions = evaluate_loader(
            raw_model,
            val_loader,
            device,
            class_weights,
            train_config.label_smoothing,
            amp_enabled,
            amp_dtype,
            threshold=None,
        )
        best_threshold_epoch = float(val_metrics["threshold"])
        score = _selection_value(val_metrics, train_config.selection_metric)
        group_lrs = {
            group.get("group_name", f"group_{index}"): float(lr)
            for index, (group, lr) in enumerate(zip(optimizer.param_groups, scheduler.get_last_lr()))
        }
        epoch_record = {
            "epoch": epoch + 1,
            "training_stage": "graph_head_warmup" if freeze_backbone else "full_finetune",
            "train_loss": running_loss / max(1, running_samples),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": scheduler.get_last_lr()[0],
            "graph_learning_rate": group_lrs.get("graph", scheduler.get_last_lr()[-1]),
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        val_predictions.to_csv(output_dir / "val_predictions_latest.csv", index=False)

        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            patience_count = 0
            best_threshold = best_threshold_epoch
            payload = _checkpoint_payload(
                raw_model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                patience_count,
                best_threshold,
                model_config,
                train_config,
                class_weight_values,
                manifest_path,
                history,
            )
            atomic_torch_save(payload, best_path)
            val_predictions.to_csv(output_dir / "val_predictions_best.csv", index=False)
        else:
            patience_count += 1

        atomic_torch_save(
            _checkpoint_payload(
                raw_model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                patience_count,
                best_threshold,
                model_config,
                train_config,
                class_weight_values,
                manifest_path,
                history,
            ),
            last_path,
        )

        print(
            f"val loss={val_metrics['loss']:.4f} | ROC-AUC={val_metrics['roc_auc']:.4f} | "
            f"PR-AUC={val_metrics['pr_auc']:.4f} | MCC={val_metrics['mcc']:.4f} | "
            f"threshold={val_metrics['threshold']:.4f}"
        )
        if patience_count >= train_config.patience:
            print(f"Early stopping after {patience_count} non-improving epochs.")
            break

    if not best_path.exists():
        raise RuntimeError("Training completed without producing a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_checkpoint["model_state"])
    best_threshold = float(best_checkpoint["validation_threshold"])

    val_metrics, val_predictions = evaluate_loader(
        raw_model,
        val_loader,
        device,
        class_weights,
        train_config.label_smoothing,
        amp_enabled,
        amp_dtype,
        threshold=best_threshold,
    )
    val_predictions.to_csv(output_dir / "val_predictions_final.csv", index=False)
    test_metrics = None
    if test_loader is not None:
        test_metrics, test_predictions = evaluate_loader(
            raw_model,
            test_loader,
            device,
            class_weights,
            train_config.label_smoothing,
            amp_enabled,
            amp_dtype,
            threshold=best_threshold,
        )
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    mixture_weights = raw_model.mixer.mixture_weights().detach().cpu().numpy()
    beta = np.stack([block.attn.beta.detach().cpu().numpy() for block in raw_model.blocks])
    np.save(output_dir / "mixture_weights.npy", mixture_weights)
    np.save(output_dir / "beta_by_layer_head.npy", beta)

    final_report = {
        "validation": val_metrics,
        "test": test_metrics,
        "validation_threshold_selected_without_test": best_threshold,
        "mixture_weights": mixture_weights,
        "beta_by_layer_head": beta,
        "resolved": resolved,
    }
    write_json(output_dir / "final_metrics.json", final_report)
    return final_report


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PhaseekV3Classifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PhaseekV3Classifier(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
