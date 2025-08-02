def coordinate():
    pt = [np.array([np.cos(i * 2 * np.pi / 20), np.sin(i * 2 * np.pi / 20), 1]) for i in range(20)]
    P = np.vstack(pt)
    V = [[pt[i] + (1 / 20) * (pt[j] - pt[i]) for j in range(20)] for i in range(20)]
    return P, V


def FEGS(sequences,start_seq,end_seq):
    P, V = coordinate()
    if "fasta" in sequences:
      sequences = [str(record.seq) for record in SeqIO.parse(sequences, "fasta")]
    sequences = sequences[start_seq:end_seq]
    l = len(sequences)

    with Pool() as pool:
        g_p = pool.starmap(GRS, [(seq, P, V) for seq in sequences])
        EL = np.array([ME(i) for i in  tqdm([g for g_list in g_p for g in g_list])]).reshape(l,158)

    char = 'ARNDCQEGHILKMFPSTWYV'
    with Pool() as pool:
        results = pool.map(SAD, [(seq, char) for seq in sequences])
        FA = np.array([res[0] for res in results])
        FD = np.array([res[1].flatten() for res in results])

    FV = np.hstack((EL, FA, FD))
    return FV


def GRS(seq, P, V):
    M = loadmat("/content/classi/M.mat")
    M = M["M"].flatten()
    l_seq = len(seq)
    k = M.shape[0]
    cha = 'ACDEFGHIKLMNPQRSTVWY'
    g = []

    for j in range(k):
        DPC = np.zeros((20, 20))
        c = [[0, 0, 0]]
        d = np.zeros(3)
        y = None

        for i in range(l_seq):
            x = np.array([seq[i] == aa for aa in M[j]]).astype(int)
            if i == 0:
                c.append(c[i] + np.dot(x, P))
            else:
                if not np.any(x):
                    d = d * (i - 1) / i
                    c.append(c[i] + [0, 0, 1] + d)
                elif not np.any(y):
                    d = d * (i - 1) / i
                    c.append(c[i] + np.dot(x, P) + d)
                else:
                    d = d * (i - 1) / i + V[int(np.where(y)[0][0])][int(np.where(x)[0][0])] / i
                    c.append(c[i] + np.dot(x, P) + d)
            y = x

        g.append(np.vstack(c))
    return g

def ME(W):
    W = W[1:, :]
    x = W.shape[0]
    D = pdist(W)
    E = squareform(D)
    sdist = np.zeros((x, x))

    for i in range(x):
        for j in range(i, x):
            if j - i == 1:
                sdist[i, j] = E[i, j]
            elif j - i > 1:
                sdist[i, j] = sdist[i, j - 1] + E[j - 1, j]

    sdist += sdist.T
    sdd = sdist + np.diag(np.ones(x))
    L = E / sdd
    ME = eigs(L, k=1)[0][0] / x
    return ME

def SAD(args):
    seq, a = args
    len_seq = len(seq)
    len_a = len(a)
    AAC = np.zeros(len_a)
    DPC = np.zeros((len_a, len_a))

    c = [np.array([s == aa for s in seq]) for aa in a]
    AAC = np.array([np.sum(c[i]) / len_seq for i in range(len_a)])

    if len_seq != 1:
        for i in range(len_a):
            for j in range(len_a):
                DPC[i, j] = np.sum((np.roll(c[j], -1) * 2 - c[i]) == 1) / (len_seq - 1)
    return AAC, DPC