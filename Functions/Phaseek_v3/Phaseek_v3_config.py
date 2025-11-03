from dataclasses import dataclass

SRC_POS = "/content/drive/MyDrive/fegs_topk_pos_unpadded"
SRC_NEG = "/content/drive/MyDrive/fegs_topk_neg_unpadded"
DST_POS = "/content/fegs_topk_pos_unpadded"
DST_NEG = "/content/fegs_topk_neg_unpadded"

TOPK_M       = 10
SEQ_LEN      = 512
N_LAYERS     = 6
D_MODEL      = 192
N_HEADS      = 6

BATCH_SIZE   = 12
NUM_WORKERS  = 2
PREFETCH     = 2
FP16_BIAS    = True

EPOCHS       = 20
LR           = 8e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC  = 0.05
LABEL_SMOOTH = 0.05

BEST_CKPT = "/content/model_graphbias_topk_best.pt"
FINAL_CKPT = "/content/model_graphbias_topk_final.pt"

@dataclass
class TrainArgs:
    src_pos: str = SRC_POS
    src_neg: str = SRC_NEG
    topk_m: int = TOPK_M
    seq_len: int = SEQ_LEN
    n_layers: int = N_LAYERS
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    prefetch: int = PREFETCH
    fp16_bias: bool = FP16_BIAS
    epochs: int = EPOCHS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup_frac: float = WARMUP_FRAC
    label_smooth: float = LABEL_SMOOTH
    best_ckpt: str = BEST_CKPT
    final_ckpt: str = FINAL_CKPT
