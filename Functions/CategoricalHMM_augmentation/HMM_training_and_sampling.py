!pip -q install hmmlearn biopython

import numpy as np
import random, math, os
from hmmlearn.hmm import MultinomialHMM
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

AMINO = "ACDEFGHIKLMNPQRSTVWY"  
AA2I = {a:i for i,a in enumerate(AMINO)}
I2AA = {i:a for a,i in AA2I.items()}

BLOSUM_NEIGHBORS = {
    'A':'AGS', 'C':'CY', 'D':'DNES', 'E':'EDNQ', 'F':'FWY', 'G':'GAS',
    'H':'HYN', 'I':'IVLM', 'K':'KRN', 'L':'LIVM', 'M':'MIL', 'N':'NDSQ',
    'P':'PAG', 'Q':'QEN', 'R':'RKHQ', 'S':'SAGTN', 'T':'TSAN', 'V':'VIL',
    'W':'WFY', 'Y':'YFW'}

def folded_mask(scores, thr=0.5):
    return [1 if s < thr else 0 for s in scores]

def folded_segments(seq, mask, min_len=5):
    segs = []
    start = None
    for i, m in enumerate(mask + [0]):  # sentinel
        if m == 1 and start is None:
            start = i
        elif (m == 0 or i == len(mask)) and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs

def concat_segments_for_training(segments):
    obs = []
    lengths = []
    for seg in segments:
        enc = [AA2I[a] for a in seg if a in AA2I]
        if enc:
            obs.extend(enc)
            lengths.append(len(enc))
    if not obs:
        raise ValueError("No valid folded segments to train on.")
    X = np.array(obs, dtype=int).reshape(-1,1)
    return X, lengths

def encode_1D(seq):
    return np.array([AA2I[a] for a in seq if a in AA2I], dtype=int).reshape(-1,1)

def _posterior_for_segment(model, seg):
    X = encode_1D(seg)
    L = len(seg)
    try:
        post = model.predict_proba(X, [L])
    except Exception:
        states = model.predict(X, [L])
        post = np.zeros((L, model.n_components))
        post[np.arange(L), states] = 1.0
    return post

def _mixture_emission(model, state_posterior):
    return state_posterior @ model.emissionprob_

def _temperature(p, tau=1.0, eps=1e-12):
    if tau == 1.0:
        s = p.sum() + eps
        return p / s
    q = np.power(p + eps, 1.0/tau)
    return q / (q.sum() + eps)

def _apply_conservative_prior(p, orig_aa, strength=1.0, eps=1e-12):
    allowed = set(BLOSUM_NEIGHBORS.get(orig_aa, orig_aa))
    mask = np.array([1.0 if (a in allowed or a==orig_aa) else (1e-6 if strength>=1 else 1.0) for a in AMINO])
    q = p * (mask ** strength)
    return q / (q.sum() + eps)

def compute_iupred_map(seqs, scorer):
    out = {}
    for i, s in enumerate(seqs):
        sc = scorer(s)        
        if len(sc) != len(s):
            raise ValueError(f"IUPred length mismatch at index {i}: seq {len(s)} vs scores {len(sc)}")
        out[i] = (s, sc)
    return out

def train_hmm_on_folded(pos_map, thr=0.5, n_states_grid=range(4,10), random_state=0, min_seg_len=5):
    folded_strs = []
    for _, (seq, scores) in pos_map.items():
        mask = folded_mask(scores, thr=thr)
        for s,e in folded_segments(seq, mask, min_len=min_seg_len):
            seg = ''.join([aa for aa in seq[s:e] if aa in AA2I])
            if seg:
                folded_strs.append(seg)

    X, lengths = concat_segments_for_training(folded_strs)

    best = None
    N = X.shape[0]
    for n in n_states_grid:
        model = MultinomialHMM(n_components=n, n_iter=200, tol=1e-3, random_state=random_state, init_params="ste")
        model.fit(X, lengths)
        k = n*(n-1) + (n-1) + n*(len(AMINO)-1)
        ll = model.score(X, lengths)
        bic = -2*ll + k*math.log(max(N,1))
        if (best is None) or (bic < best['bic']):
            best = {'n': n, 'model': model, 'bic': bic, 'll': ll}
    return best['model'], best


def hmm_guided_mutate_sequence(seq, scores, model,thr=0.5, rate=0.03, temperature=0.9,conservative_strength=0.0, avoid_identity=True):
    out = list(seq)
    mask = folded_mask(scores, thr=thr)
    runs = folded_segments(seq, mask, min_len=1)

    for (s,e) in runs:
      seg = ''.join(out[s:e])
      post = _posterior_for_segment(model, seg)   
      mixed = _mixture_emission(model, post)       

      for t in range(e - s):
          idx = s + t
          if out[idx] not in AA2I:
              continue
          if random.random() >= rate:
              continue

          p = mixed[t].copy()
          if conservative_strength > 0:
              p = _apply_conservative_prior(p, out[idx], strength=conservative_strength)
          p = _temperature(p, tau=temperature)

          if avoid_identity:
              p[AA2I[out[idx]]] = 0.0
              ssum = p.sum()
              if ssum <= 0:
                  p = mixed[t].copy()
              else:
                  p = p / ssum

          new_aa = np.random.choice(list(AMINO), p=p)
          out[idx] = new_aa

    return ''.join(out)

def augment_dataset_hmm_guided(pos_map, model,n_copies=3, thr=0.5, rate=0.03,temperature=0.9, conservative_strength=0.5,avoid_identity=True):
    augmented = []
    for idx, (seq, scores) in pos_map.items():
        for _ in range(n_copies):
            aug = hmm_guided_mutate_sequence(
                seq, scores, model,
                thr=thr, rate=rate, temperature=temperature,
                conservative_strength=conservative_strength,
                avoid_identity=avoid_identity
            )
            augmented.append(aug)
    return augmented


np.random.seed(48)
random.seed(48)

pos_map = compute_iupred_map(pos, aiupred)

hmm, model_info = train_hmm_on_folded(
    pos_map,
    thr=0.5,
    n_states_grid=range(4,10),
    random_state=42,
    min_seg_len=5
)
print(f"Chosen n_states: {model_info['n']} | BIC: {model_info['bic']:.2f}")

aug_sequences = augment_dataset_hmm_guided(
    pos_map, hmm,
    n_copies=10,        
    thr=0.5,
    rate=0.05,          
    temperature=0.9,     
    conservative_strength=0.5,  
    avoid_identity=True
)
print(f"Augmented sequences generated: {len(aug_sequences)} (from {len(pos)} originals)")

out_path = "/content/training_pos_aug_hmm_foldedonly.fasta"
records = []
count = 0
for i, seq in enumerate(aug_sequences):
    count += 1
    records.append(SeqRecord(Seq(seq), id=f"pos_aug_{i+1}", description="hmm_guided_folded_only"))
SeqIO.write(records, out_path, "fasta")
print("Wrote:", out_path)
