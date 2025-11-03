import math, os
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from .Phaseek_v3 import TransformerClassifier, Config
from .Phaseek_v3_data import make_dataloaders
from . import config as CFG

def build_model(vocab_size: int, args: CFG.TrainArgs, device: str) -> TransformerClassifier:
    model_cfg = Config(
        vocab_size=vocab_size,
        block_size=args.seq_len,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.d_model,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        causal=False,
        use_graph_bias=True
    )
    model = TransformerClassifier(model_cfg, topk_m=args.topk_m, label_smooth=args.label_smooth, weight_decay=args.weight_decay).to(device)
    return model

def make_scheduler(optimizer, total_train_steps: int, warmup_frac: float):
    warmup_steps = max(10, int(warmup_frac * total_train_steps))
    def lr_lambda(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_train_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    tot_loss, tot_n, correct = 0.0, 0, 0
    all_probs, all_labels = [], []
    for xb_cpu, yb_cpu, Lhat_cpu in loader:
        xb   = xb_cpu.to(device, non_blocking=True)
        yb   = yb_cpu.to(device, non_blocking=True)
        Lhat = Lhat_cpu.to(device, non_blocking=True)

        logits, loss = model(xb, yb, Lhat_stack=Lhat)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        tot_loss += float(loss.item()) * xb.size(0)
        tot_n    += xb.size(0)
        correct  += int((logits.argmax(dim=-1) == yb).sum().item())
        all_probs.append(probs.cpu()); all_labels.append(yb.cpu())
    avg_loss = tot_loss / max(1, tot_n)
    acc = correct / max(1, tot_n)

    auc = prauc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        auc   = roc_auc_score(labels_np, probs_np)
        prauc = average_precision_score(labels_np, probs_np)
    except Exception:
        pass

    model.train()
    return avg_loss, acc, auc, prauc

def split_data(X_seq, paths, y):
    X_tr, X_va, p_tr, p_va, y_tr, y_va = train_test_split(
        X_seq, paths, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_tr, X_va, p_tr, p_va, y_tr, y_va

def fit(
    X_seq: np.ndarray, paths: np.ndarray, y: np.ndarray,
    args: CFG.TrainArgs, device: str
):
    # split
    Xseq_tr, Xseq_va, paths_tr, paths_va, y_tr, y_va = split_data(X_seq, paths, y)
    vocab_size = int(max(Xseq_tr.max(initial=0), Xseq_va.max(initial=0))) + 1

    # dataloaders (configure globals used in collate)
    from . import data as DataMod
    DataMod.TOPK_M = args.topk_m
    DataMod.SEQ_LEN = args.seq_len
    DataMod.FP16_BIAS = args.fp16_bias
    train_loader, val_loader = make_dataloaders(
        Xseq_tr, Xseq_va, paths_tr, paths_va, y_tr, y_va,
        args.batch_size, args.num_workers, args.prefetch, device
    )

    # model/opt/sched
    model = build_model(vocab_size, args, device)
    optimizer = model.configure_optimizers(lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_train_steps = args.epochs * max(1, math.ceil(len(train_loader)))
    scheduler = make_scheduler(optimizer, total_train_steps, args.warmup_frac)
    amp_enabled = (device == "cuda")
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_val = float('inf'); patience = 6; pat = 0
    for epoch in range(args.epochs):
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for xb_cpu, yb_cpu, Lhat_cpu in pbar:
            xb   = xb_cpu.to(device, non_blocking=True)
            yb   = yb_cpu.to(device, non_blocking=True)
            Lhat = Lhat_cpu.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits, loss = model(xb, yb, Lhat_stack=Lhat)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_acc, val_auc, val_prauc = evaluate(model, val_loader, device)
        msg = f"  -> val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        if val_auc   is not None: msg += f" | val_auc={val_auc:.4f}"
        if val_prauc is not None: msg += f" | val_pr_auc={val_prauc:.4f}"
        print(msg)

        if val_loss < best_val - 1e-4:
            best_val = val_loss; pat = 0
            torch.save(model.state_dict(), args.best_ckpt)
            print(f"  Saved checkpoint -> {args.best_ckpt}")
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping triggered."); break

    if os.path.exists(args.best_ckpt):
        model.load_state_dict(torch.load(args.best_ckpt, map_location=device))
    final_loss, final_acc, final_auc, final_prauc = evaluate(model, val_loader, device)
    print("\n=== Final (best checkpoint) ===")
    print(
        f"val_loss={final_loss:.4f} | val_acc={final_acc:.4f}"
        + (f" | val_auc={final_auc:.4f}" if final_auc is not None else "")
        + (f" | val_pr_auc={final_prauc:.4f}" if final_prauc is not None else "")
    )

    print("\nLearned beta per block (graph influence):")
    for i, blk in enumerate(model.transformer.h):
        print(f"  block {i}: beta={blk.attn.beta.detach().cpu().item():.4f}")

    with torch.no_grad():
        logits = model.mixer.alpha.unsqueeze(0) + model.mixer.delta    # [H,m]
        pi = torch.softmax(logits / model.mixer.tau, dim=-1).cpu().numpy()
    print("\nShared alpha logits:", model.mixer.alpha.detach().cpu().numpy())
    print("Head-specific mixture weights π (rows=heads, cols=top-k motifs):")
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    print(pi)

    torch.save(model.state_dict(), args.final_ckpt)
    print(f"\nSaved final weights -> {args.final_ckpt}")
