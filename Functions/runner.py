import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import torch
from classifier import Transformer
from XGBoost import XGBoost
from Configue import CfgNode
from FEGS_feature_extraction import FEGSFeatureExtractor

warnings.filterwarnings("ignore")

try:
    import random
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass

clf = XGBoost.XGM()

AA20 = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {aa: i for i, aa in enumerate(AA20)}

def _to_device_tensor(arr, device):
    if arr is None:
        return None
    t = torch.tensor(arr, dtype=torch.float32)
    return t.to(device)

def build_pair_bias_from_FEGS_SAD(seq: str) -> np.ndarray:

    AAC, DPC = FEGSFeatureExtractor._SAD_static((seq, AA20))
    T = len(seq)
    P = np.zeros((T, T), dtype=np.float32)
    idxs = np.array([AA_INDEX[ch] for ch in seq], dtype=int)
    P = DPC[idxs][:, idxs] 
    P = 0.5 * (P + P.T)
    return P

def slice_bias_for_window(bias_full, start, win):

    if bias_full is None:
        return None
    if bias_full.ndim == 2:
        return bias_full[start:start+win, start:start+win]
    elif bias_full.ndim == 3:
        return bias_full[:, start:start+win, start:start+win]
    else:
        raise ValueError("bias_full must be (T,T) or (H,T,T)")

def edit(sequence: str) -> str:
    return ''.join([char.upper() for char in sequence if char in "ACDEFGHIKLMNPQRSTVWY"])

def enh(scores):
    k = 50
    scores = np.asarray(scores, dtype=float)
    weights = np.array([1.0 / (1.0 + np.exp(-k * (2.0 * s - 1.0))) for s in scores])
    return float(np.sum(scores * weights) / np.sum(weights))

def padd(sequence_values):
    seq = list(sequence_values)
    return seq + [0.0] * (5537 - len(seq))

def SCORE(P, U):
    def scorer(seq_vals, u_val):
        vec = np.array(padd(seq_vals) + [u_val], dtype=float).reshape(1, 5538)
        prob = float(clf.predict_proba(vec)[:, 1][0])
        truncated = [v for v in vec.ravel() if v != 0.0]
        return 0.3 * vec[0, -1] + 0.4 * enh(truncated[:-1]) + 0.3 * prob

    if len(P) > 5537:
        chunks = [P[i * 5537:(i + 1) * 5537] for i in range(len(P) // 5537 + 1)]
        chunks = [c for c in chunks if len(c) > 0]
        return float(enh([scorer(c, U) for c in chunks]))
    else:
        return float(scorer(P, U))

def d(x, u):
    if u > 0.7:
        return x * np.exp(-1.2 * (x - u))
    else:
        return x if x > 0.7 else x

def Score1(index, scores, L, n):
    i = index - 1
    if i <= L:
        return sum(scores[:i + 1]) / (i + 1)
    elif L + 1 <= i < n - L:
        return sum(scores[i - L:i + 1]) / (L + 1)
    else:
        return sum(scores[i - L:n - L + 1]) / (n - i + 1)

def predict_full_sequence_with_optional_bias(model: Transformer, sequence: str, bias_full_np=None):

    T = len(sequence)
    if T <= 512:
        bm = _to_device_tensor(bias_full_np[None, ...], model.device) if bias_full_np is not None else None
        batch = model.Encode([sequence])
        out = model._classifier.predict_proba(batch, "sig", bm).cpu().numpy()
        return float(out.reshape(-1)[0])
    preds = []
    for start in range(T - 512 + 1):
        window_seq = sequence[start:start + 512]
        x = model.Encode([window_seq])  

        if bias_full_np is not None:
            bias_slice = slice_bias_for_window(bias_full_np, start, 512)
            if bias_slice.ndim == 2:
                bias_tensor = _to_device_tensor(bias_slice[None, ...], model.device) 
            else:
                bias_tensor = _to_device_tensor(bias_slice[None, ...], model.device) 
        else:
            bias_tensor = None

        p = model._classifier.predict_proba(x, "sig", bias_tensor).cpu().numpy()
        preds.append(p[0, 0])

    return float(enh(np.array(preds)))

def SW(sequence_chunks, model: Transformer):

    if len(sequence_chunks) > 700:
        chunks = [sequence_chunks[i * 700:(i + 1) * 700] for i in range(len(sequence_chunks) // 700 + 1)]
        chunks = [c for c in chunks if len(c) > 0]
        return [model.predict_proba(c) for c in chunks]
    else:
        return model.predict_proba(sequence_chunks)


def main():
    parser = argparse.ArgumentParser(description="LLPS Analysis Script (with optional FEGS bias)")
    parser.add_argument("--sequence", type=str, help="Protein sequence or .fasta path.")
    parser.add_argument("--id", type=str, help="Protein ID.")
    parser.add_argument("--directory", type=str, help="Output directory.")
    parser.add_argument("--end_sequence", type=int, default=500, help="Endpoint for FASTA mode.")
    parser.add_argument("--plot", type=bool, default=True, help="Plot flag (kept for compatibility).")

    parser.add_argument("--use_bias", action="store_true",
                        help="If set, derives a (T,T) bias prior from FEGS SAD/DPC and feeds it into attention.")

    args = parser.parse_args()

    SequenceArg = args.sequence
    End_sequence = args.end_sequence
    directory = args.directory or "DefaultDir"
    ID = args.id or "DefaultID"
    model = Transformer("c", use_graph_bias= False)

    base_res = os.path.join(os.path.dirname(__file__), "../Results")
    path_dir = os.path.join(base_res, str(directory))
    path_id = os.path.join(path_dir, str(ID))
    os.makedirs(path_id, exist_ok=True)

    if SequenceArg and SequenceArg != "" and ".fasta" not in SequenceArg:
        Sequence = edit(SequenceArg)
        n = len(Sequence)
        k = max(5, min(50, int(np.ceil(0.1 * n))))
        L = k // 2
        S = [Sequence[i:i + L] for i in range(n - L + 1)]
        Sc = SW(S, model)
        Sc1 = np.concatenate(Sc) if isinstance(Sc, list) else (Sc if isinstance(Sc, np.ndarray) else np.array(Sc))
        bias_np = build_pair_bias_from_FEGS_SAD(Sequence) if args.use_bias else None
        u = predict_full_sequence_with_optional_bias(model, Sequence, bias_full_np=bias_np)
        scores = [d(Score1(x, Sc1, L, n), u) for x in range(1, n + 1)]
        scores = [float(s[0]) if isinstance(s, (list, np.ndarray)) else float(s) for s in scores]
        final_score = SCORE(scores, u)

        pd.DataFrame({"scores": scores, "seq": list(Sequence)}).to_csv(
            os.path.join(path_id, "scores.csv"), index=False)
        print(f"Score: {final_score}")
    elif SequenceArg and os.path.exists(SequenceArg) and ".fasta" in SequenceArg:
        from Bio import SeqIO
        fasta_data = [(str(rec.id), edit(str(rec.seq))) for rec in SeqIO.parse(SequenceArg, "fasta")]
        fasta_data = fasta_data[:End_sequence]

        results = {
            "id": [i[0] for i in fasta_data],
            "seq": [i[1] for i in fasta_data],
            "LLPS_score": [],
            "Residue-level score": []
        }

        for rec_id, sequence in tqdm(fasta_data):
            try:
                n = len(sequence)
                k = max(5, min(50, int(np.ceil(0.1 * n))))
                L = k // 2
                S = [sequence[i:i + L] for i in range(n - L + 1)]
                Sc = SW(S, model)
                Sc1 = np.concatenate(Sc) if isinstance(Sc, list) else (Sc if isinstance(Sc, np.ndarray) else np.array(Sc))
                bias_np = build_pair_bias_from_FEGS_SAD(sequence) if args.use_bias else None

                u = predict_full_sequence_with_optional_bias(model, sequence, bias_full_np=bias_np)

                scores = [d(Score1(x, Sc1, L, n), u) for x in range(1, n + 1)]
                scores = [float(s[0]) if isinstance(s, (list, np.ndarray)) else float(s) for s in scores]
                final_score = SCORE(scores, u)

                results["LLPS_score"].append(final_score)
                results["Residue-level score"].append(scores)
            except Exception:
                results["LLPS_score"].append(None)
                results["Residue-level score"].append(None)

        out_csv = os.path.join(path_id, "LLPS_prediction_of_seqs.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("Analysis complete.")
    else:
        print("No valid sequence or FASTA file provided.")

if __name__ == "__main__":
    main()
