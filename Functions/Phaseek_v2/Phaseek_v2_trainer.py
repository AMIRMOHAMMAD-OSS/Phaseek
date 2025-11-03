import os, math, numpy as np, torch
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from .Phaseek_v2_metrics import evaluate
from .Phaseek_v2 import TransformerClassifier, Config

def split_data(X_seq, X_feat, y):
    return train_test_split(X_seq, X_feat, y, test_size=0.2, random_state=42, stratify=y)

def make_scheduler(optimizer, total_train_steps: int, warmup_frac: float):
    warmup_steps = max(10, int(warmup_frac * total_train_steps))
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_train_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def fit(
    X_seq, X_feat, y,
    seq_len: int, num_feat: int,
    n_layers: int, n_heads: int, d_model: int,
    epochs: int, batch_size: int, num_workers: int, pin_memory: bool,
    lr: float, weight_decay: float, warmup_frac: float, label_smooth: float,
    best_ckpt: str, final_ckpt: str, device: str, sklearn_ok: bool
):
    # split
    Xseq_tr, Xseq_va, Xf_tr, Xf_va, y_tr, y_va = split_data(X_seq, X_feat, y)

    # vocab size
    vocab_size = int(max(Xseq_tr.max(initial=0), Xseq_va.max(initial=0))) + 1

    # loaders
    from .data import make_loaders
    train_loader, val_loader = make_loaders(
        Xseq_tr, Xseq_va, Xf_tr, Xf_va, y_tr, y_va,
        batch_size, num_workers, pin_memory
    )

    # model
    model_cfg = Config(
        vocab_size=vocab_size,
        block_size=seq_len,
        n_layer=n_layers,
        n_head=n_heads,
        n_embd=d_model,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        causal=False
    )
    model = TransformerClassifier(model_cfg, num_feat=num_feat, seq_len=seq_len, label_smooth=label_smooth).to(device)
    optimizer = model.configure_optimizers(lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    total_train_steps = epochs * max(1, math.ceil(len(train_loader)))
    scheduler = make_scheduler(optimizer, total_train_steps, warmup_frac)
    scaler = GradScaler("cuda", enabled=(device=="cuda"))

    # train
    best_val = float('inf'); patience = 5; pat = 0
    print("\n=== Training (fast, on-GPU bias) ===")
    from tqdm import tqdm
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for xb_cpu, xfeat_cpu, yb_cpu in pbar:
            xb   = xb_cpu.to(device, non_blocking=True)
            xfea = xfeat_cpu.to(device, non_blocking=True)
            yb   = yb_cpu.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                _, loss = model(xb, xfea, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_acc, val_auc, val_prauc = evaluate(model, val_loader, device, sklearn_ok)
        msg = f"  -> val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        if val_auc   is not None: msg += f" | val_auc={val_auc:.4f}"
        if val_prauc is not None: msg += f" | val_pr_auc={val_prauc:.4f}"
        print(msg)

        if val_loss < best_val - 1e-4:
            best_val = val_loss; pat = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  Saved checkpoint -> {best_ckpt}")
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping triggered."); break

    # final
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    from .metrics import evaluate as eval_fn
    final_loss, final_acc, final_auc, final_prauc, curves = eval_fn(model, val_loader, device, sklearn_ok, return_curves=True)
    print("\n=== Final (best checkpoint) ===")
    print(f"val_loss={final_loss:.4f} | val_acc={final_acc:.4f}"
          + (f" | val_auc={final_auc:.4f}" if final_auc is not None else "")
          + (f" | val_pr_auc={final_prauc:.4f}" if final_prauc is not None else ""))

    if sklearn_ok and curves is not None:
        prec, rec, thr, labels_np, probs_np = curves
        f1s = (2*prec*rec)/(prec+rec+1e-12)
        if len(thr) > 0:
            import numpy as np
            best_i = f1s[:-1].argmax()
            best_thr = thr[best_i]
            yhat = (probs_np >= best_thr).astype(np.int64)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels_np, yhat)
            print(f"\nBest F1={f1s[best_i]:.4f} at threshold={best_thr:.4f}")
            print(f"Precision={prec[best_i]:.4f} | Recall={rec[best_i]:.4f}")
            print("Confusion matrix @best F1:\n", cm)

    print("\nLearned beta per block:")
    for i, blk in enumerate(model.transformer.h):
        print(f"  block {i}: beta={blk.attn.beta.detach().cpu().item():.4f}")

    torch.save(model.state_dict(), final_ckpt)
    print(f"\nSaved final weights -> {final_ckpt}")
