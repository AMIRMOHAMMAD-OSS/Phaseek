
import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from Bio import SeqIO


class FEGSFeatureExtractor:

    def __init__(self, m_mat_path: str = "/content/classi/M.mat", processes: int | None = None):
        self.m_mat_path = m_mat_path
        self.processes = processes
        mat = loadmat(self.m_mat_path)
        self.M = mat["M"].flatten()
        self.P, self.V = self._coordinate()
        self.num_M = len(self.M)

    @staticmethod
    def _coordinate():
        pt = [np.array([np.cos(i * 2 * np.pi / 20), np.sin(i * 2 * np.pi / 20), 1.0]) for i in range(20)]
        P = np.vstack(pt) 
        V = [[pt[i] + (1.0 / 20.0) * (pt[j] - pt[i]) for j in range(20)] for i in range(20)]
        return P, V

    @staticmethod
    def _GRS_static(seq: str, P: np.ndarray, V: list[list[np.ndarray]], M_flat):

        l_seq = len(seq)
        k = len(M_flat)
        g = []

        for j in range(k):
            c = [np.array([0.0, 0.0, 0.0])]
            d = np.zeros(3, dtype=float)
            y = None

            motif = M_flat[j]
            for i in range(l_seq):
                x = np.array([seq[i] == aa for aa in motif], dtype=int)
                if i == 0:
                    c.append(c[i] + x @ P)  
                else:
                    if not np.any(x):
                        d = d * (i - 1) / i
                        c.append(c[i] + np.array([0.0, 0.0, 1.0]) + d)
                    elif not np.any(y):
                        d = d * (i - 1) / i
                        c.append(c[i] + x @ P + d)
                    else:
                        prev_idx = int(np.where(y)[0][0])
                        curr_idx = int(np.where(x)[0][0])
                        d = d * (i - 1) / i + V[prev_idx][curr_idx] / i
                        c.append(c[i] + x @ P + d)
                y = x

            g.append(np.vstack(c))  
        return g

    @staticmethod
    def _ME_static(W: np.ndarray) -> float:

        W = W[1:, :]  
        x = W.shape[0]
        D = pdist(W)
        E = squareform(D)
        sdist = np.zeros((x, x), dtype=float)

        for i in range(x):
            for j in range(i, x):
                if j - i == 1:
                    sdist[i, j] = E[i, j]
                elif j - i > 1:
                    sdist[i, j] = sdist[i, j - 1] + E[j - 1, j]

        sdist += sdist.T
        sdd = sdist + np.diag(np.ones(x))
        L = E / sdd
        val = eigs(L, k=1)[0][0]
        return float(np.real(val) / x)

    @staticmethod
    def _SAD_static(args):
        seq, a = args
        len_seq = len(seq)
        len_a = len(a)

        c = [np.array([s == aa for s in seq], dtype=bool) for aa in a]
        AAC = np.array([np.sum(c[i]) / max(len_seq, 1) for i in range(len_a)], dtype=float)

        DPC = np.zeros((len_a, len_a), dtype=float)
        if len_seq > 1:
            for i in range(len_a):
                for j in range(len_a):

                    DPC[i, j] = np.sum((np.roll(c[j], -1).astype(int) * 2 - c[i].astype(int)) == 1) / (len_seq - 1)

        return AAC, DPC

    @staticmethod
    def _load_sequences_any(sequences, start_seq: int, end_seq: int | None):

        if isinstance(sequences, str) and "fasta" in sequences.lower():
            seqs = [str(record.seq) for record in SeqIO.parse(sequences, "fasta")]
        elif isinstance(sequences, (list, tuple, np.ndarray)):
            seqs = list(sequences)
        else:
            raise ValueError("`sequences` must be a FASTA path or a list/array of sequences.")

        return seqs[start_seq:end_seq]

    def extract(self, sequences, start_seq: int = 0, end_seq: int | None = None) -> np.ndarray:

        P, V, M_flat = self.P, self.V, self.M
        sequences = self._load_sequences_any(sequences, start_seq, end_seq)
        l = len(sequences)
        with Pool(processes=self.processes) as pool:
            g_p = pool.starmap(self._GRS_static, [(seq, P, V, M_flat) for seq in sequences])

        EL_vals = [self._ME_static(W) for g_list in tqdm(g_p, desc="ME", leave=False) for W in g_list]
        EL = np.array(EL_vals, dtype=float).reshape(l, self.num_M)

        char = 'ARNDCQEGHILKMFPSTWYV' 
        with Pool(processes=self.processes) as pool:
            results = pool.map(self._SAD_static, [(seq, char) for seq in sequences])

        FA = np.array([res[0] for res in results], dtype=float)               
        FD = np.array([res[1].flatten() for res in results], dtype=float)      

        FV = np.hstack((EL, FA, FD))
        return FV
