import torch

@torch.no_grad()
def evaluate(model, loader, device: str, sklearn_ok: bool, return_curves: bool=False):
    from typing import Optional, Tuple
    model.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    all_probs, all_labels = [], []
    for xb_cpu, xfeat_cpu, yb_cpu in loader:
        xb   = xb_cpu.to(device, non_blocking=True)
        xfea = xfeat_cpu.to(device, non_blocking=True)
        yb   = yb_cpu.to(device, non_blocking=True)
        logits, loss = model(xb, xfea, yb)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        total_loss += float(loss.item()) * xb.size(0); total_n += xb.size(0)
        correct += int((logits.argmax(dim=-1) == yb).sum().item())
        all_probs.append(probs.detach().cpu()); all_labels.append(yb.detach().cpu())
    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    auc = prauc = None
    curves = None
    if sklearn_ok:
        import numpy as np
        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
            auc   = roc_auc_score(labels_np, probs_np)
            prauc = average_precision_score(labels_np, probs_np)
            if return_curves:
                prec, rec, thr = precision_recall_curve(labels_np, probs_np)
                curves = (prec, rec, thr, labels_np, probs_np)
        except Exception:
            pass
    model.train()
    if return_curves:
        return avg_loss, acc, auc, prauc, curves
    return avg_loss, acc, auc, prauc
