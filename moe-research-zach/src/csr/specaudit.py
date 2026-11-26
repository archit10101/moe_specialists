"""specialization-audit-v1 helpers (docs/SPECIALIZATION_AUDIT_V1_PREREG.md). Pure numpy/torch; no model code.

Data units (windows, halves, strata), deterministic permutations and bootstrap, in-project implementations of
k-means, TF-IDF, randomized SVD, multinomial logistic regression and the Hungarian assignment (scikit-learn/SciPy
are not installed), and the routing / functional metric kernels.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

SEED = 20261101
WIN = 128
MIN_PARTIAL = 64
MAX_WIN_PER_DOC = 4
N_AUDIT_DOCS = 8192
N_ROW_GROUPS = 12
SUPPORT_WINDOWS = 32
N_PERM = 1000
N_BOOT = 2000
HALF_SALT = "specaudit-v1-half"
CAP_SALT = "specaudit-v1-cap"
CHUNK_BINS = [1, 64, 128, 192]  # [1,64),[64,128),[128,192),[192,inf)


# ------------------------------------------------------------------ data units
def choose_audit_row_groups(n_total: int, excluded):
    ex = set(int(x) for x in excluded)
    eligible = sorted(i for i in range(n_total) if i not in ex)
    rng = np.random.default_rng(SEED)
    return sorted(int(x) for x in rng.choice(eligible, size=N_ROW_GROUPS, replace=False)), len(eligible)


def select_documents(eligible_ids_sorted, n=N_AUDIT_DOCS):
    if len(eligible_ids_sorted) < n:
        raise ValueError(f"only {len(eligible_ids_sorted)} eligible audit documents < {n}: STOP")
    perm = np.random.default_rng(SEED).permutation(len(eligible_ids_sorted))
    return sorted(eligible_ids_sorted[i] for i in perm[:n])


def unit_hash(salt: str, key: str) -> float:
    return int.from_bytes(hashlib.sha256((salt + key).encode()).digest()[:8], "little") / 2**64


def half_of(doc_id: str) -> int:
    """0 = discovery, 1 = effect-estimation."""
    return 0 if unit_hash(HALF_SALT, doc_id) < 0.5 else 1


def candidate_windows(L: int):
    """Prediction positions 0..L-2; [start, end) windows of 128, final partial kept if >= 64 positions."""
    n_pos = L - 1
    out = [(a, min(a + WIN, n_pos)) for a in range(0, max(n_pos, 0), WIN)]
    if out and out[-1][1] - out[-1][0] < MIN_PARTIAL:
        out = out[:-1]
    return out


def select_windows(L: int):
    """Returns list of (window_index, start, end) for at most 4 windows spread across the document."""
    cand = candidate_windows(L)
    n = len(cand)
    if n <= MAX_WIN_PER_DOC:
        idx = list(range(n))
    else:
        idx = sorted(set(int(math.floor(x + 0.5)) for x in np.linspace(0, n - 1, MAX_WIN_PER_DOC)))
    return [(i, cand[i][0], cand[i][1]) for i in idx]


def doc_length_quartile_edges(lengths):
    return np.quantile(np.asarray(lengths, float), [0.25, 0.5, 0.75]).tolist()


def bin_of(x, edges):
    return np.searchsorted(np.asarray(edges), x, side="right")


def window_strata(win_doc_len, win_len, qedges):
    """Stratum id = doc-length quartile (0..3) * 2 + window type (0 full, 1 partial)."""
    return bin_of(np.asarray(win_doc_len), qedges) * 2 + (np.asarray(win_len) < WIN).astype(int)


# ------------------------------------------------------------------ permutations / bootstrap
def stratified_permutations(strata_by_half, n_perm=N_PERM, seed=SEED):
    """strata_by_half: int array per window; permutations are generated within (stratum) groups. The caller passes
    half*16+stratum so halves are permuted separately. Returns (n_perm, n_windows) int32 index arrays: label of
    window w under permutation p is labels[perm[p, w]]."""
    s = np.asarray(strata_by_half)
    rng = np.random.default_rng(seed)
    groups = [np.nonzero(s == g)[0] for g in np.unique(s)]
    out = np.empty((n_perm, len(s)), np.int32)
    for p in range(n_perm):
        for g in groups:
            out[p, g] = g[rng.permutation(len(g))]
    return out


def bootstrap_counts(n_docs, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_docs) multiplicities; same construction as csr.train_lib.paired_bootstrap (integers)."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_boot, n_docs), np.float32)
    for b in range(n_boot):
        M[b] = np.bincount(rng.integers(0, n_docs, n_docs), minlength=n_docs)
    return M


def ci95(x):
    lo, hi = np.percentile(np.asarray(x), [2.5, 97.5])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------ information measures
def entropy(p, axis=None):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)


def mutual_information(joint):
    """joint: (..., K, E) non-negative mass; returns MI in nats over the last two axes."""
    J = np.asarray(joint, float)
    tot = J.sum(axis=(-2, -1), keepdims=True)
    P = J / np.where(tot > 0, tot, 1)
    pk, pe = P.sum(-1, keepdims=True), P.sum(-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * (np.log(P) - np.log(pk) - np.log(pe)), 0.0)
    return t.sum(axis=(-2, -1))


def nmi(joint):
    P = np.asarray(joint, float)
    P = P / P.sum()
    hk, he = entropy(P.sum(1)), entropy(P.sum(0))
    return float(mutual_information(P) / ((hk + he) / 2)) if hk + he > 0 else 0.0


def jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(-1)
        b = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(-1)
    return 0.5 * (a + b)


def weighted_pairwise_cosine(X, m):
    """Exact activation-weighted mean cosine over distinct pairs: X (W, D) L2-normalized, m (W,) >= 0."""
    m = np.asarray(m, float)
    s = (m[:, None] * X).sum(0)
    num = float(s @ s - np.sum(m * m * np.sum(X * X, 1)))
    den = float(m.sum() ** 2 - np.sum(m * m))
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------ Hungarian (min-cost assignment)
def hungarian(cost):
    """Min-cost perfect assignment for a square matrix (Jonker-Volgenant style O(n^3) shortest augmenting path).
    Returns col_of_row (n,)."""
    C = np.asarray(cost, float)
    n = C.shape[0]
    assert C.shape == (n, n)
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p = np.zeros(n + 1, int)  # p[j] = row matched to column j (1-based), 0 = none
    way = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_of_row = np.zeros(n, int)
    for j in range(1, n + 1):
        col_of_row[p[j] - 1] = j - 1
    return col_of_row


# ------------------------------------------------------------------ k-means (deterministic)
def kmeans(X, K, n_init=20, seed=SEED, max_iter=300, device="cpu"):
    """k-means++ init (init i seeded with seed+i), Lloyd until assignments unchanged; empty cluster re-seeded with
    the point farthest from its centre. Returns (centroids (K,D) float32, labels, inertia, best_init)."""
    import torch
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=device)
    n = Xt.shape[0]
    x2 = (Xt * Xt).sum(1)
    best = None
    for i in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        first = int(torch.randint(n, (1,), generator=g))
        C = [Xt[first]]
        d2 = ((Xt - C[0]) ** 2).sum(1).clamp_min(0)
        for _ in range(1, K):
            pr = (d2 / d2.sum()).double().cpu()
            idx = int(torch.multinomial(pr, 1, generator=g))
            C.append(Xt[idx])
            d2 = torch.minimum(d2, ((Xt - Xt[idx]) ** 2).sum(1).clamp_min(0))
        C = torch.stack(C)
        lab = None
        for _ in range(max_iter):
            dist = x2[:, None] - 2 * Xt @ C.T + (C * C).sum(1)[None]
            new = dist.argmin(1)
            if lab is not None and torch.equal(new, lab):
                break
            lab = new
            cnt = torch.bincount(lab, minlength=K)
            C = torch.zeros_like(C).index_add_(0, lab, Xt) / cnt.clamp_min(1)[:, None].to(Xt.dtype)
            for k in torch.nonzero(cnt == 0).flatten().tolist():
                dmin = dist.gather(1, lab[:, None]).squeeze(1)
                far = int(dmin.argmax())
                C[k] = Xt[far]
                lab = lab.clone()
                lab[far] = k
        dist = x2[:, None] - 2 * Xt @ C.T + (C * C).sum(1)[None]
        lab = dist.argmin(1)
        inertia = float(dist.gather(1, lab[:, None]).clamp_min(0).sum())
        if best is None or inertia < best[2] - 1e-9 * abs(best[2]):
            best = (C.cpu().numpy().astype(np.float32), lab.cpu().numpy(), inertia, i)
    return best


def assign_nearest(X, C):
    X, C = np.asarray(X, np.float32), np.asarray(C, np.float32)
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None]
    return d.argmin(1)


# ------------------------------------------------------------------ TF-IDF + randomized SVD
_TOK = re.compile(r"(?u)\b\w\w+\b")


def tfidf_terms(text):
    w = _TOK.findall(text.lower())
    return w + [w[i] + " " + w[i + 1] for i in range(len(w) - 1)]


def tfidf_fit(texts, max_features=100_000):
    tf = Counter()
    df = Counter()
    for t in texts:
        c = Counter(tfidf_terms(t))
        tf.update(c)
        df.update(c.keys())
    vocab_terms = sorted(tf, key=lambda k: (-tf[k], k))[:max_features]
    vocab = {k: i for i, k in enumerate(sorted(vocab_terms))}
    n = len(texts)
    idf = np.array([math.log((1 + n) / (1 + df[k])) + 1.0 for k in sorted(vocab_terms)])
    return vocab, idf


def tfidf_transform(texts, vocab, idf):
    """Returns CSR pieces (indptr, indices, data) of row-L2-normalized tf-idf."""
    indptr, indices, data = [0], [], []
    for t in texts:
        c = Counter(x for x in tfidf_terms(t) if x in vocab)
        cols = np.array(sorted(vocab[k] for k in c), np.int64)
        inv = {vocab[k]: v for k, v in c.items()}
        vals = np.array([inv[j] * idf[j] for j in cols], float)
        nrm = np.sqrt((vals ** 2).sum())
        if nrm > 0:
            vals = vals / nrm
        indices.extend(cols.tolist())
        data.extend(vals.tolist())
        indptr.append(len(indices))
    return np.array(indptr, np.int64), np.array(indices, np.int64), np.array(data, np.float64)


def _sparse(indptr, indices, data, ncol):
    import torch
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return torch.sparse_coo_tensor(np.vstack([rows, indices]), data, (len(indptr) - 1, ncol)).coalesce()


def randomized_svd_fit(A_csr, ncol, k=256, n_oversamples=10, n_iter=5, seed=SEED):
    """Halko randomized range finder with QR-normalized power iterations; returns V (ncol, k) float64."""
    import torch
    A = _sparse(*A_csr, ncol)
    g = torch.Generator().manual_seed(seed)
    Om = torch.randn(ncol, k + n_oversamples, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.sparse.mm(A, Om))
    for _ in range(n_iter):
        Z, _ = torch.linalg.qr(torch.sparse.mm(A.t(), Q))
        Q, _ = torch.linalg.qr(torch.sparse.mm(A, Z))
    B = torch.sparse.mm(A.t(), Q).t()  # (k+p, ncol)
    _, _, Vh = torch.linalg.svd(B, full_matrices=False)
    V = Vh[:k].t()
    # deterministic sign: largest-|.| entry of each component positive
    sgn = torch.sign(V.gather(0, V.abs().argmax(0, keepdim=True)))
    return (V * sgn).numpy()


def svd_project(A_csr, ncol, V):
    import torch
    Z = torch.sparse.mm(_sparse(*A_csr, ncol), torch.as_tensor(V)).numpy()
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return (Z / np.where(n > 0, n, 1)).astype(np.float32)


# ------------------------------------------------------------------ multinomial logistic regression (C = 1)
def logreg_fit(X, y, n_classes, C=1.0, max_iter=500, tol=1e-6):
    """Minimizes sum_i CE_i + ||W||^2/(2C) (intercept unpenalized), float64 L-BFGS from zero. Returns (W, b, info)."""
    import torch
    Xt = torch.as_tensor(np.asarray(X, np.float64))
    yt = torch.as_tensor(np.asarray(y, np.int64))
    W = torch.zeros(Xt.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="sum") + (W * W).sum() / (2 * C)
        loss.backward()
        return loss
    opt.step(closure)
    loss = closure()
    gn = float(torch.cat([W.grad.flatten(), b.grad.flatten()]).abs().max())
    return W.detach().numpy(), b.detach().numpy(), {"loss": float(loss), "max_abs_grad": gn}


def logreg_predict(X, W, b):
    return np.argmax(np.asarray(X, np.float64) @ W + b, axis=1)


def macro_f1(y, yhat, classes=None):
    y, yhat = np.asarray(y), np.asarray(yhat)
    classes = np.unique(y) if classes is None else classes
    f = []
    for k in classes:
        tp = np.sum((yhat == k) & (y == k))
        fp = np.sum((yhat == k) & (y != k))
        fn = np.sum((yhat != k) & (y == k))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0)
    return float(np.mean(f))


def balanced_accuracy(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean([np.mean(yhat[y == k] == k) for k in np.unique(y)]))


# ------------------------------------------------------------------ functional interaction kernel
def decile_codes(x):
    edges = np.quantile(np.asarray(x, float), np.arange(1, 10) / 10)
    return np.searchsorted(edges, x, side="right")


def cell_keys(ce, gate, pos, chunk_bin, doc_bin):
    """Cell id per token from five stratifiers (deciles over the given tokens + fixed bins)."""
    a = decile_codes(ce)
    b = decile_codes(gate)
    c = decile_codes(pos)
    return (((a * 10 + b) * 10 + c) * 4 + np.asarray(chunk_bin)) * 4 + np.asarray(doc_bin)


def interaction(delta, is_top, cells, windows, support=SUPPORT_WINDOWS):
    """ATT-style stratified interaction for one expert. Returns dict with estimate (or None if insufficient)."""
    delta, is_top, cells, windows = map(np.asarray, (delta, is_top, cells, windows))
    uc, inv = np.unique(cells, return_inverse=True)
    nT = np.bincount(inv, weights=is_top, minlength=len(uc))
    nA = np.bincount(inv, minlength=len(uc)).astype(float)
    nO = nA - nT
    sT = np.bincount(inv, weights=delta * is_top, minlength=len(uc))
    sO = np.bincount(inv, weights=delta, minlength=len(uc)) - sT
    ok = (nT > 0) & (nO > 0)
    tok_ok = ok[inv]
    wT = len(np.unique(windows[tok_ok & (is_top == 1)]))
    wO = len(np.unique(windows[tok_ok & (is_top == 0)]))
    res = {"n_T": int(is_top.sum()), "n_O": int(len(is_top) - is_top.sum()),
           "n_T_matched": int(nT[ok].sum()), "n_O_matched": int(nO[ok].sum()),
           "windows_T_matched": int(wT), "windows_O_matched": int(wO)}
    if wT < support or wO < support:
        res["estimate"] = None
        return res
    pi = nT[ok] / nT[ok].sum()
    mT, mO = sT[ok] / nT[ok], sO[ok] / nO[ok]
    res.update({"estimate": float(np.sum(pi * (mT - mO))), "mean_T": float(np.sum(pi * mT)),
                "mean_O_matched": float(np.sum(pi * mO))})
    return res


# ------------------------------------------------------------------ routing joint distributions
def normalized_gate_mass(topi, gates, E):
    """topi (N, k) int, gates (N, k) -> (N, E) normalized routed-expert gate mass (rows sum to 1)."""
    topi = np.asarray(topi, np.int64)
    g = np.asarray(gates, np.float64)
    g = g / g.sum(1, keepdims=True)
    M = np.zeros((len(topi), E))
    np.put_along_axis(M, topi, g, axis=1)
    return M


def window_mass(topi, gates, win_of_token, n_windows, E):
    """Per-window summed normalized gate mass (n_windows, E)."""
    M = normalized_gate_mass(topi, gates, E)
    out = np.zeros((n_windows, E))
    np.add.at(out, np.asarray(win_of_token), M)
    return out


def joint_from_window_mass(wmass, labels, K):
    """Expected joint P(topic, expert) from per-window masses and window labels; normalized to sum 1."""
    J = np.zeros((K, wmass.shape[1]))
    np.add.at(J, np.asarray(labels), wmass)
    return J / J.sum()


def hungarian_fast(cost):
    """Same algorithm as `hungarian` with the inner column loops vectorized (identical optimum)."""
    C = np.asarray(cost, float)
    n = C.shape[0]
    INF = np.inf
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p = np.zeros(n + 1, int)
    way = np.zeros(n + 1, int)
    Cp = np.zeros((n + 1, n + 1))
    Cp[1:, 1:] = C
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used
            free[0] = False
            cur = Cp[i0] - u[i0] - v
            upd = free & (cur < minv)
            minv[upd] = cur[upd]
            way[upd] = j0
            cand = np.where(free, minv, INF)
            j1 = int(np.argmin(cand))
            delta = cand[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_of_row = np.zeros(n, int)
    col_of_row[p[1:] - 1] = np.arange(n)
    return col_of_row
