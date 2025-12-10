"""
Methods:
1. SeqProp (Gradient-based, continuous optimization with annealing).
2. AutoRegressive (Greedy/Stochastic step-by-step design).
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from classifier_fgs import transformer 


# --------------------------
# Config / globals
# --------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda_amp = (device.type == "cuda") and torch.cuda.is_available() 

AA = "ACDEFGHIKLMNPQRSTVWY"
K = len(AA)

# Initialize the model globally for the scoring functions
try:
    model = transformer("c") # Default mode initialization
except Exception as e:
    print(f"Warning: Could not initialize global model (transformer('c')). Scoring and generation functions will fail if run independently. Error: {e}")
    model = None 



def get_classifier_and_W_AA(mode: str = "c"):

    wrapper = transformer(mode)   
    clf = wrapper._classifier
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    vocab = wrapper.vocab_dict

    aa_token_ids = []
    for a in AA:
        if a not in vocab:
            raise ValueError(f"Amino acid '{a}' not found in tokenizer vocabulary.")
        aa_token_ids.append(vocab[a])

    W_full = clf.transformer.wte.weight  # [vocab_size, d_emb]
    W_AA = W_full[aa_token_ids, :].detach().to(device)  # [20, d_emb]

    clf.to(device)
    return wrapper, clf, W_AA


# --------------------------
# Scoring and Auto-Regressive Design Functions
# --------------------------

def scorerr(seq):
    """Clean the sequence and score it using the global model wrapper."""
    if model is None:
        raise RuntimeError("Model not initialized. Cannot run scorerr.")

    def edit(i):
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join(c.upper() for c in i if c.upper() in valid_chars)
    
    Sequence = edit(seq)
    if not Sequence:
        return 0.0

    u = model.predict_proba([Sequence])
    
   
    if u.ndim == 2 and u.shape[1] > 0:
        u = u[0, 0]
    elif u.ndim == 1:
        u = u[0]
    else:
        
        return 0.0
        
    score = float(u)
    return score


def run_autoregressive_design(length: int = 120,
                              k_choices: int = 3,
                              start_seq: str = "MM",
                              mode: str = "c"):
    """
    Generates a sequence using a greedy/stochastic auto-regressive approach.
    At each step, selects the next residue from the top k_choices that maximize score.
    """
    if model is None:
        global model
        try:
             model = transformer(mode)
        except Exception as e:
            raise RuntimeError(f"Model initialization failed for AutoRegressive mode: {e}")

    CChars = AA
    p11 = list(start_seq)
    
    while len(p11) < 3 and len(p11) < length:
         p11.append(np.random.choice(list(CChars)))

    print(f"--- Starting AutoRegressive Design (L={length}, k={k_choices}) ---")
    
    pbar = tqdm(range(len(p11), length), desc="Generating Sequence", leave=False)
    for _ in pbar:

        current_seqs = ["".join(p11) + i for i in CChars]
        SSC = model.predict_proba(current_seqs).ravel().tolist()
        
        OO = {i: SSC[CChars.index(i)] for i in CChars}
        top_k = sorted(OO, key=lambda x: -OO[x])[:k_choices]
        next_res = np.random.choice(top_k)
        p11.append(next_res)
        current_score = OO[next_res]
        pbar.set_postfix(Score=f"{current_score:.4f}", Res=next_res)

    final_seq = "".join(p11[:length])
    final_score = scorerr(final_seq)
    
    print(f"\nAutoRegressive Final Sequence (L={len(final_seq)}):")
    print(final_seq)
    print(f"Predicted LLPS Score: {final_score:.6f}")
    
    return final_seq, final_score


def greedy_refine(seq, wrapper, max_iters=200):
    """
    Simple local search on discrete sequence under wrapper.predict_proba.
    Tries single-residue mutations that improve the score.
    """
    best_seq = list(seq)
    best_score = scorerr("".join(best_seq)) 

    for _ in range(max_iters):
        improved = False
        for pos in range(len(best_seq)):
            current_res = best_seq[pos]
            for a in AA:
                if a == current_res:
                    continue
                cand = best_seq[:]
                cand[pos] = a
                s = scorerr("".join(cand)) # Use scorerr
                if s > best_score:
                    best_score = s
                    best_seq = cand
                    improved = True
                    # restart local search from new best
                    break
            if improved:
                break
        if not improved:
            break

    return "".join(best_seq), best_score


# --------------------------
# SeqProp Optimization Loop (run_seqprop - with Temperature Annealing)
# --------------------------

def run_seqprop(length: int = 120,
                steps: int = 300,
                lr: float = 0.1,  
                entropy_weight: float = 0.1, 
                mode: str = "c",
                seed: int = 0,
                log_every: int = 20,
                refine: bool = True,
                temp_start: float = 2.0,
                temp_end: float = 0.1):

    torch.manual_seed(seed)
    np.random.seed(seed)

    wrapper, clf, W_AA = get_classifier_and_W_AA(mode=mode)

    L = length
    theta = torch.randn(L, K, device=device, requires_grad=True)  
    
    optimizer = torch.optim.Adam([theta], lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    best_theta = None
    best_score_val = -1.0

    print(f"--- Starting SeqProp Optimization (L={L}, Steps={steps}, Mode={mode}) ---")
    print(f"Hyperparameters: LR={lr}, EW={entropy_weight}, Temp={temp_start} -> {temp_end}")


    for step in range(steps):
        current_temp = temp_start + (temp_end - temp_start) * (step / steps)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_cuda_amp):
            P_soft = F.softmax(theta / current_temp, dim=-1)
            E = P_soft @ W_AA
            
            # Forward pass (using cloned logic from seqprop_forward)
            x = E.unsqueeze(0)
            B, T, C = x.size()
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = clf.transformer.wpe(pos)
            x = clf.transformer.drop(x + pos_emb)
            for block in clf.transformer.h:
                x = block(x)
            x = clf.transformer.ln_f(x)
            logits = clf.ny(clf.classifier_head(x)).mean(1)
            score = torch.sigmoid(logits)[:, 0]

            P_real = F.softmax(theta, dim=-1)
            entropy = -(P_real * torch.clamp(P_real, min=1e-9).log()).sum(-1).mean()
            loss = -score.mean() + (entropy_weight * entropy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        score_val = float(score.item())
        
        if score_val > best_score_val and step > (steps // 2):
            best_score_val = score_val
            best_theta = theta.detach().clone()

        if (step + 1) % max(1, log_every) == 0 or step == 0:
            print(
                f"Step {step+1}/{steps} | "
                f"Temp: {current_temp:.2f} | "
                f"Score (soft): {score_val:.4f} | "
                f"Entropy: {float(entropy.item()):.4f}"
            )

    if best_theta is None:
        best_theta = theta.detach()

    # Final Decode
    print("\n==============================")
    with torch.no_grad():
        P_final = F.softmax(best_theta, dim=-1).cpu().numpy()
        idxs = np.argmax(P_final, axis=1)
        seq_seqprop = "".join(AA[i] for i in idxs)
        score_seqprop = scorerr(seq_seqprop)

    print("SeqProp-relaxed design (argmax decode):")
    print(seq_seqprop)
    print(f"Predicted LLPS score (wrapper) before refinement: {score_seqprop:.6f}")

    if refine:
        print("Running refinement...")
        # Note: refine uses the global scorerr, so 'wrapper' argument is mostly decorative
        refined_seq, refined_score = greedy_refine(seq_seqprop, wrapper, max_iters=200)
        print("\nRefined sequence:")
        print(refined_seq)
        print(f"Final Score: {refined_score:.6f}")
        final_seq, final_score = refined_seq, refined_score
    else:
        final_seq, final_score = seq_seqprop, score_seqprop

    print("==============================\n")
    return final_seq, final_score


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLPS sequence design using SeqProp or AutoRegressive methods."
    )
    # New argument to select method
    parser.add_argument("--method", type=str, default="seqprop", choices=["seqprop", "autoregressive"],
                        help="Design method: 'seqprop' (gradient) or 'autoregressive' (greedy).")
    
    # Common arguments
    parser.add_argument("--length", type=int, default=120, help="Sequence length to design.")
    parser.add_argument("--mode", type=str, default="c", help="Transformer mode (a–h).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_seqs", type=int, default=1, help="Number of sequences to generate.")

    # SeqProp specific arguments
    parser.add_argument("--steps", type=int, default=300, help="SeqProp: Number of gradient steps.")
    parser.add_argument("--lr", type=float, default=0.02, help="SeqProp: Learning rate for Adam.")
    parser.add_argument("--entropy_weight", type=float, default=0.05, help="SeqProp: Weight for entropy penalty.")
    parser.add_argument("--log_every", type=int, default=20, help="SeqProp: Log every N steps.")
    parser.add_argument("--no_refine", action="store_true", help="Disable greedy refinement step after SeqProp.")
    parser.add_argument("--temp_start", type=float, default=2.0, help="SeqProp: Starting temperature for annealing.")
    parser.add_argument("--temp_end", type=float, default=0.1, help="SeqProp: Ending temperature for annealing.")
    
    # AutoRegressive specific arguments
    parser.add_argument("--k_choices", type=int, default=3, help="AutoRegressive: Number of top residues to randomly choose from.")
    parser.add_argument("--start_seq", type=str, default="M", help="AutoRegressive: Starting sequence (e.g., 'MM').")

    args = parser.parse_args()
    
    results = []
    
    if args.method == "seqprop":
        print(f"\n--- Running SeqProp for {args.num_seqs} Sequence(s) ---\n")
        for i in range(args.num_seqs):
            print(f"================== SEQUENCE {i+1}/{args.num_seqs} (Method: SeqProp) ==================")
            current_seed = args.seed + i
            
            final_seq, final_score = run_seqprop(
                length=args.length, steps=args.steps, lr=args.lr, entropy_weight=args.entropy_weight,
                mode=args.mode, seed=current_seed, log_every=args.log_every, 
                refine=not args.no_refine, temp_start=args.temp_start, temp_end=args.temp_end
            )
            results.append((final_seq, final_score))
            print(f"FINAL DESIGN {i+1} (SeqProp): Score: {final_score:.6f}")

    elif args.method == "autoregressive":
        print(f"\n--- Running AutoRegressive Design (k={args.k_choices}) for {args.num_seqs} Sequence(s) ---\n")
        
        global model
        try:
            model = transformer(args.mode)
        except Exception as e:
            print(f"Error initializing model for AutoRegressive mode: {e}")
            return
            
        for i in range(args.num_seqs):
            print(f"================== SEQUENCE {i+1}/{args.num_seqs} (Method: AutoRegressive) ==================")
            np.random.seed(args.seed + i) 
            
            final_seq, final_score = run_autoregressive_design(
                length=args.length, k_choices=args.k_choices, start_seq=args.start_seq, mode=args.mode
            )

            results.append((final_seq, final_score))
            print(f"FINAL DESIGN {i+1} (AutoRegressive): Score: {final_score:.6f}")

if __name__ == "__main__":

    main()
