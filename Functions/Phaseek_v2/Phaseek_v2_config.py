from dataclasses import dataclass

# Default paths (override via CLI flags)
POS_FEATS_CSV = "/content/drive/MyDrive/pos_feature_extracted_Oversampled_dataset.csv"
NEG_FEATS_CSV = "/content/drive/MyDrive/neg_feature_extracted_dataset.csv"
NUM_FEAT   = 158
SEQ_LEN    = 512
N_LAYERS   = 6
N_HEADS    = 6
D_MODEL    = 192
EPOCHS       = 50
BATCH_SIZE   = 32
NUM_WORKERS  = 2
PIN_MEMORY   = True
LR           = 8e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC  = 0.05
LABEL_SMOOTH = 0.05

BEST_CKPT  = "/content/model_graph_bias_best.pt"
FINAL_CKPT = "/content/model_graph_bias_final.pt"

@dataclass
class TrainArgs:
    pos_feats_csv: str = POS_FEATS_CSV
    neg_feats_csv: str = NEG_FEATS_CSV
    num_feat: int = NUM_FEAT
    seq_len: int = SEQ_LEN
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    d_model: int = D_MODEL
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup_frac: float = WARMUP_FRAC
    label_smooth: float = LABEL_SMOOTH
    best_ckpt: str = BEST_CKPT
    final_ckpt: str = FINAL_CKPT
