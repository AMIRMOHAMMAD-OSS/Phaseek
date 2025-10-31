model_cfg = Config(
    vocab_size=vocab_size,
    block_size=SEQ_LEN,
    n_layer=6,
    n_head=6,
    n_embd=192,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
    causal=False,
    use_graph_bias=True
)

model = TransformerClassifier(model_cfg).to(device)
optimizer = model.configure_optimizers(lr=8e-4, betas=(0.9, 0.95), weight_decay=0.1)
EPOCHS = 50
total_train_steps = EPOCHS * max(1, math.ceil(len(train_loader)))
warmup_steps = max(10, int(0.05 * total_train_steps))
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_train_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
amp_enabled = (device == "cuda")
scaler = GradScaler("cuda", enabled=amp_enabled)
def evaluate(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    correct = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb_cpu, yb_cpu, bb_cpu in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)
            bb = bb_cpu.to(device, non_blocking=True)
            logits, loss = model(xb, yb, bias_matrix=bb)
            probs = torch.softmax(logits, dim=-1)[:, 1]  
            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            all_probs.append(probs.detach().cpu())
            all_labels.append(yb.detach().cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    auc = prauc = None
    if SKLEARN_OK:
        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc   = roc_auc_score(labels_np, probs_np)
            prauc = average_precision_score(labels_np, probs_np)
        except Exception:
            pass
    model.train()
    return avg_loss, acc, auc, prauc


best_val = float('inf')
patience = 5
pat = 0
best_path = "/content/model_graph_bias_best.pt"

print("\n=== Training ===")
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
    for xb_cpu, yb_cpu, bb_cpu in pbar:
        xb = xb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)
        bb = bb_cpu.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits, loss = model(xb, yb, bias_matrix=bb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    val_loss, val_acc, val_auc, val_prauc = evaluate(model, val_loader)
    msg = f"  -> val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
    if val_auc  is not None: msg += f" | val_auc={val_auc:.4f}"
    if val_prauc is not None: msg += f" | val_pr_auc={val_prauc:.4f}"
    print(msg)

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        pat = 0
        torch.save(model.state_dict(), best_path)
        print(f"  Saved checkpoint -> {best_path}")
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping triggered.")
            break

if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))
final_loss, final_acc, final_auc, final_prauc = evaluate(model, val_loader)
print("\n=== Final (best checkpoint) ===")
print(f"val_loss={final_loss:.4f} | val_acc={final_acc:.4f}" +
      (f" | val_auc={final_auc:.4f}" if final_auc is not None else "") +
      (f" | val_pr_auc={final_prauc:.4f}" if final_prauc is not None else ""))

if SKLEARN_OK:
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for xb_cpu, yb_cpu, bb_cpu in val_loader:
            xb = xb_cpu.to(device); yb = yb_cpu.to(device); bb = bb_cpu.to(device)
            logits, _ = model(xb, None, bias_matrix=bb)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.append(probs.cpu()); labels_list.append(yb.cpu())
    probs = torch.cat(probs_list).numpy()
    labels = torch.cat(labels_list).numpy()

    from sklearn.metrics import precision_recall_curve, f1_score
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1s = (2*prec*rec)/(prec+rec+1e-12)
    best_i = f1s[:-1].argmax()
    best_thr = thr[best_i]
    yhat = (probs >= best_thr).astype(np.int64)
    cm = confusion_matrix(labels, yhat)
    print(f"\nBest F1={f1s[best_i]:.4f} at threshold={best_thr:.4f}")
    print(f"Precision={prec[best_i]:.4f} | Recall={rec[best_i]:.4f}")
    print("Confusion matrix @best F1:\n", cm)

print("\nLearned beta per block:")
for i, blk in enumerate(model.transformer.h):
    print(f"  block {i}: beta={blk.attn.beta.detach().cpu().item():.4f}")

final_path = "/content/model_graph_bias_final.pt"
torch.save(model.state_dict(), final_path)
print(f"\nSaved final weights -> {final_path}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
