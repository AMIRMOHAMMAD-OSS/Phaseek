import argparse, numpy as np, pandas as pd, os
from fegs_fast import config as CFG
from fegs_fast.utils import set_seed, setup_device
from fegs_fast.data import load_feature_tables
from fegs_fast.trainer import fit

from user_hooks import get_sequences 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pos_feats_csv", type=str, default=CFG.POS_FEATS_CSV)
    p.add_argument("--neg_feats_csv", type=str, default=CFG.NEG_FEATS_CSV)
    p.add_argument("--num_feat", type=int, default=CFG.NUM_FEAT)
    p.add_argument("--seq_len", type=int, default=CFG.SEQ_LEN)
    p.add_argument("--n_layers", type=int, default=CFG.N_LAYERS)
    p.add_argument("--n_heads", type=int, default=CFG.N_HEADS)
    p.add_argument("--d_model", type=int, default=CFG.D_MODEL)
    p.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    p.add_argument("--pin_memory", action="store_true", default=CFG.PIN_MEMORY)
    p.add_argument("--lr", type=float, default=CFG.LR)
    p.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY)
    p.add_argument("--warmup_frac", type=float, default=CFG.WARMUP_FRAC)
    p.add_argument("--label_smooth", type=float, default=CFG.LABEL_SMOOTH)
    p.add_argument("--best_ckpt", type=str, default=CFG.BEST_CKPT)
    p.add_argument("--final_ckpt", type=str, default=CFG.FINAL_CKPT)
    args = p.parse_args()

    set_seed(42)
    device = setup_device()
    pos_feats, neg_feats = load_feature_tables(args.pos_feats_csv, args.neg_feats_csv, args.num_feat,
                                               pos_rows=659, neg_rows=4859)
    print(f"pos_features: {pos_feats.shape} | neg_features: {neg_feats.shape}")
    pos_seq, neg_seq = get_sequences(seq_len=args.seq_len)
    print(f"pos_seq: {pos_seq.shape} | neg_seq: {neg_seq.shape}")
    assert pos_feats.shape[0] == pos_seq.shape[0] and neg_feats.shape[0] == neg_seq.shape[0], \
        "Mismatch: features vs sequences row counts"
    assert pos_seq.shape[1] == args.seq_len and neg_seq.shape[1] == args.seq_len, \
        "Mismatch: sequence length vs SEQ_LEN"
    X_seq  = np.vstack([pos_seq,  neg_seq])
    X_feat = np.vstack([pos_feats, neg_feats])
    y      = np.concatenate([np.ones(len(pos_seq), dtype=np.int64),
                             np.zeros(len(neg_seq), dtype=np.int64)], axis=0)
    print(f"Total: {len(y)} | Pos: {y.sum()} | Neg: {(y==0).sum()}")
    try:
        import sklearn  
        SKLEARN_OK = True
    except Exception:
        SKLEARN_OK = False

    fit(
        X_seq=X_seq, X_feat=X_feat, y=y,
        seq_len=args.seq_len, num_feat=args.num_feat,
        n_layers=args.n_layers, n_heads=args.n_heads, d_model=args.d_model,
        epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
        lr=args.lr, weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, label_smooth=args.label_smooth,
        best_ckpt=args.best_ckpt, final_ckpt=args.final_ckpt, device=device, sklearn_ok=SKLEARN_OK
    )

if __name__ == "__main__":
    main()
