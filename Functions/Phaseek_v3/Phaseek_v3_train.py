import argparse
import numpy as np
import torch
from fegs import config as CFG
from fegs.utils import set_seed, setup_device, list_npz_sorted
from fegs.trainer import fit
from user_hooks import get_sequences_and_paths 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_pos", type=str, default=CFG.SRC_POS)
    p.add_argument("--src_neg", type=str, default=CFG.SRC_NEG)
    p.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p.add_argument("--seq_len", type=int, default=CFG.SEQ_LEN)
    p.add_argument("--topk_m", type=int, default=CFG.TOPK_M)
    p.add_argument("--n_layers", type=int, default=CFG.N_LAYERS)
    p.add_argument("--d_model", type=int, default=CFG.D_MODEL)
    p.add_argument("--n_heads", type=int, default=CFG.N_HEADS)
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    p.add_argument("--prefetch", type=int, default=CFG.PREFETCH)
    p.add_argument("--lr", type=float, default=CFG.LR)
    p.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--warmup_frac", type=float, default=CFG.WARMUP_FRAC)
    p.add_argument("--label_smooth", type=float, default=CFG.LABEL_SMOOTH)
    p.add_argument("--fp16_bias", action="store_true", default=CFG.FP16_BIAS)
    p.add_argument("--best_ckpt", type=str, default=CFG.BEST_CKPT)
    p.add_argument("--final_ckpt", type=str, default=CFG.FINAL_CKPT)
    args = p.parse_args()

    set_seed(42)
    device = setup_device()

    pos_npz = list_npz_sorted(args.src_pos)
    neg_npz = list_npz_sorted(args.src_neg)

    print("Loading encoded sequences from your hooks...")
    pos_seq, neg_seq = get_sequences_and_paths(seq_len=args.seq_len)  # returns np arrays
    assert len(pos_seq) == len(pos_npz), "Mismatch: #pos sequences vs #pos .npz files"
    assert len(neg_seq) == len(neg_npz), "Mismatch: #neg sequences vs #neg .npz files"

    X_seq  = np.vstack([pos_seq, neg_seq])
    paths  = np.array(pos_npz + neg_npz, dtype=object)
    y      = np.concatenate([np.ones(len(pos_seq), dtype=np.int64),
                             np.zeros(len(neg_seq), dtype=np.int64)], axis=0)
    print(f"Total samples: {len(y)} | Pos: {y.sum()} | Neg: {(y==0).sum()}")

    train_args = CFG.TrainArgs(
        src_pos=args.src_pos, src_neg=args.src_neg,
        topk_m=args.topk_m, seq_len=args.seq_len,
        n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads,
        batch_size=args.batch_size, num_workers=args.num_workers, prefetch=args.prefetch,
        fp16_bias=args.fp16_bias, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
        label_smooth=args.label_smooth, best_ckpt=args.best_ckpt, final_ckpt=args.final_ckpt
    )

    fit(X_seq=X_seq, paths=paths, y=y, args=train_args, device=device)

if __name__ == "__main__":
    main()
