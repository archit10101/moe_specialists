"""Stage `cluster`: label-free paper clusters and every group table the experiments use.

Follows the original pipeline (gc_lib.py, gc_discover.py, sem_tables.py) except where noted:
1. Each paper is cut into up to ``max_views`` views of ``view_tokens`` tokens spread over the whole paper
   (gc_lib.view_starts); papers shorter than one view are embedded whole (the original left them unlabelled).
2. Views are embedded with a frozen encoder (bge-small-en-v1.5, CLS pooling, L2-normalised; MiniLM / tf-idf
   for B7) and z-scored per dimension with the statistics of the fit papers.
3. Centroids: spherical k-means with Sinkhorn-balanced assignments, fit on base-split papers (the original
   trained a SwAV-style prototype network with batch Sinkhorn targets; RECONSTRUCTION.md).
4. q = softmax(mean over a paper's views of cos(view, centroid) / assign_temp), not balanced at assignment;
   ``hard`` = argmax q; ``soft`` = q; ``sharp`` (Semantic C6) = q**sharp_power renormalised.
5. Shuffled controls permute whole rows across papers within each split and within paper-length quantile
   bins (original orv1_tables.py), so in every split each shuffled cluster has exactly as many papers as its
   semantic twin and about the same token mass, but random content (doc §2.5).

Writes ``$WORK/groups/<table>.npz`` (hard, soft?, k) and ``$WORK/groups/summary.json``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..config import Profile
from ..data.loader import load_docs, load_meta
from ..data.prepare import Tokenizer
from ..utils import data_dir, device, groups_dir, is_done, log, mark_done, write_json
from ..eval.metrics import nmi

ENCODERS = {"bge": ("BAAI/bge-small-en-v1.5", "", 512, "cls"),
            "e5": ("intfloat/e5-base-v2", "passage: ", 512, "mean"),
            "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", 256, "mean")}
SHUFFLE_SEED = 999


# ------------------------------------------------------------------------------ texts

def view_starts(n: int, v: int, max_views: int = 4) -> list[int]:
    """Starts of up to max_views non-overlapping v-token views separated by >= v tokens, spread over the
    paper (original gc_lib.view_starts). A paper shorter than one view gets one view of the whole paper."""
    if n <= v:
        return [0]
    nv = min(max_views, (n + v) // (2 * v))
    if nv <= 1:
        return [(n - v) // 2]
    return [int(round(i * (n - v) / (nv - 1))) for i in range(nv)]


def paper_views(p: Profile) -> tuple[list[str], np.ndarray]:
    """Decoded views of every paper in global paper-id order: (texts, owner paper id of each view)."""
    meta, docs = load_meta(p), load_docs(p)
    tok = Tokenizer(meta["tokenizer"])
    v = p.cluster.view_tokens
    texts, owner, pid = [], [], 0
    for s_idx, s in enumerate(meta["split_order"]):
        stream = np.memmap(data_dir(p) / f"{s}.tokens.bin", dtype=np.uint16, mode="r")
        lens = docs["n_tokens"][docs["split"] == s_idx]
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for st, n in zip(starts, lens):
            for o in view_starts(int(n), v, p.cluster.max_views):
                texts.append(tok.decode(stream[st + o: st + min(n, o + v)]))
                owner.append(pid)
            pid += 1
    assert pid == meta["n_docs"]
    return texts, np.asarray(owner, np.int64)


# ------------------------------------------------------------------------------ embeddings

@torch.no_grad()
def embed(texts: list[str], embedder: str, dim_tfidf: int = 256) -> np.ndarray:
    if embedder == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(max_features=50_000, sublinear_tf=True, min_df=2).fit_transform(texts)
        k = min(dim_tfidf, X.shape[1] - 1, X.shape[0] - 1)
        E = TruncatedSVD(k, random_state=0).fit_transform(X)
        return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    from transformers import AutoModel, AutoTokenizer
    name, prefix, max_len, pooling = ENCODERS[embedder]
    dev = device()
    tk = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(dev).eval()
    if dev.type == "cuda":
        model = model.half()
    out = []
    bs = 256 if dev.type == "cuda" else 16
    for i in range(0, len(texts), bs):
        enc = tk([prefix + t for t in texts[i:i + bs]], max_length=max_len, truncation=True,
                 padding=True, return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state.float()
        if pooling == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        if (i // bs) % 100 == 0:
            log(f"  {embedder}: embedded {i + len(texts[i:i + bs]):,}/{len(texts):,}")
    return torch.cat(out).numpy().astype(np.float32)


# ------------------------------------------------------------------------------ balanced k-means

def sinkhorn(scores: torch.Tensor, temp: float, iters: int) -> torch.Tensor:
    """Balanced soft assignment: rows sum to 1, columns sum to n/K. scores [n, K] (cosines)."""
    n, k = scores.shape
    logq = scores.double() / temp
    log_r, log_c = -math.log(n), -math.log(k)        # row mass 1/n, column mass 1/K
    u = torch.zeros(n, dtype=torch.float64)
    v = torch.zeros(k, dtype=torch.float64)
    for _ in range(iters):
        u = log_r - torch.logsumexp(logq + v[None], dim=1)
        v = log_c - torch.logsumexp(logq + u[:, None], dim=0)
    q = torch.exp(logq + u[:, None] + v[None])
    return (q / q.sum(1, keepdim=True)).float()


def fit_centroids(X: np.ndarray, k: int, iters: int, sk_iters: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X)
    # k-means++ seeding on cosine distance
    C = [Xt[torch.randint(len(Xt), (1,), generator=g)].squeeze(0)]
    for _ in range(1, k):
        d = (1 - Xt @ torch.stack(C).T).clamp_min(0).min(1).values
        C.append(Xt[torch.multinomial(d / d.sum() + 1e-12, 1, generator=g)].squeeze(0))
    C = torch.stack(C)
    for _ in range(iters):
        q = sinkhorn(Xt @ C.T, temp=0.05, iters=sk_iters)
        hard = q.argmax(1)
        newC = torch.zeros_like(C).index_add_(0, hard, Xt)
        empty = torch.bincount(hard, minlength=k) == 0
        newC[empty] = C[empty]
        C = torch.nn.functional.normalize(newC, dim=-1)
    return C


def paper_mean(X: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, X.shape[1]), np.float64)
    np.add.at(out, owner, X)
    return out / np.maximum(1, np.bincount(owner, minlength=n))[:, None]


def cluster_tables(Xv: np.ndarray, owner: np.ndarray, n: int, base_mask: np.ndarray, k: int, p: Profile) -> dict:
    """Xv [views, dim] L2-normalised view embeddings; owner [views] paper id. Returns hard / soft / sharp."""
    cc = p.cluster
    rng = np.random.default_rng(0)
    fit_idx = np.flatnonzero(base_mask)
    fit_idx = rng.choice(fit_idx, min(cc.fit_docs, len(fit_idx)), replace=False)
    fit_views = np.isin(owner, fit_idx)
    mu, sd = Xv[fit_views].mean(0), Xv[fit_views].std(0) + 1e-6         # z-score on the fit papers
    Z = (Xv - mu) / sd
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    P = paper_mean(Z, owner, n)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    C = fit_centroids(P[fit_idx].astype(np.float32), k, cc.kmeans_iters, cc.sinkhorn_iters)
    logits = paper_mean(Z @ C.numpy().T.astype(np.float64), owner, n) / cc.assign_temp
    q = np.exp(logits - logits.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    sharp = q ** cc.sharp_power
    sharp /= sharp.sum(1, keepdims=True)
    return dict(hard=q.argmax(1), soft=q.astype(np.float32), sharp=sharp.astype(np.float32))


# ------------------------------------------------------------------------------ stage

def save_table(p: Profile, name: str, hard, k, soft=None) -> None:
    d = groups_dir(p)
    d.mkdir(parents=True, exist_ok=True)
    kw = dict(hard=np.asarray(hard, np.int16), k=np.int64(k))
    if soft is not None:
        kw["soft"] = np.asarray(soft, np.float16)
    np.savez(d / f"{name}.npz", **kw)


def within_split_permutation(split: np.ndarray, mass: np.ndarray | None = None, bins: int = 1,
                             seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Permutation of paper ids that only swaps papers inside the same split and, if ``mass`` is given,
    inside the same paper-length quantile bin of that split (original orv1_tables.py: 20 bins). A shuffled
    table then has exactly its semantic twin's paper counts per cluster in every split (the bank split the
    specialists train on, the val split they are scored on) and about the same token mass."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(split))
    for s in np.unique(split):
        ix = np.flatnonzero(split == s)
        if mass is None or bins <= 1:
            groups = [ix]
        else:
            edges = np.quantile(mass[ix], np.linspace(0, 1, bins + 1)[1:-1])
            b = np.searchsorted(edges, mass[ix], side="right")
            groups = [ix[b == j] for j in np.unique(b)]
        for g in groups:
            perm[g] = g[rng.permutation(len(g))]
    return perm


def build(p: Profile) -> None:
    out = groups_dir(p)
    if is_done(out, "cluster"):
        log(f"group tables already built in {out}")
        return
    meta, docs = load_meta(p), load_docs(p)
    n = meta["n_docs"]
    base_mask = docs["split"] == meta["split_order"].index("base")
    perm = within_split_permutation(docs["split"], docs["n_tokens"], p.cluster.shuffle_bins)
    assert len(perm) == n
    summary = {}

    dom = docs["domain"].astype(np.int64)
    save_table(p, "domain", dom, len(meta["domains"]))
    save_table(p, "domain_shuf", dom[perm], len(meta["domains"]))

    log("reading paper views")
    texts, owner = paper_views(p)
    log(f"  {len(texts):,} views of {n:,} papers")
    emb_cache = {}

    def emb(name):
        if name not in emb_cache:
            log(f"embedding {len(texts):,} views with {name}")
            emb_cache[name] = embed(texts, name)
            np.save(out / f"emb_{name}.npy", emb_cache[name].astype(np.float16))
        return emb_cache[name]

    def record(name, hard, k):
        sizes = np.bincount(hard, minlength=k)
        summary[name] = dict(k=k, nmi_domain=nmi(hard, dom, k, len(meta["domains"])),
                             min_size_frac=float(sizes.min() / sizes.mean()),
                             max_size_frac=float(sizes.max() / sizes.mean()))
        log(f"  {name}: K={k} NMI(domain)={summary[name]['nmi_domain']:.3f} "
            f"size range {summary[name]['min_size_frac']:.2f}-{summary[name]['max_size_frac']:.2f} x mean")

    cc = p.cluster
    for k in sorted({cc.main_k, *cc.alt_ks}):
        t = cluster_tables(emb(cc.embedder), owner, n, base_mask, k, p)
        names = ["sem", "sem_sharp"] if k == cc.main_k else []
        if k in cc.alt_ks:
            names.append(f"sem_k{k}")
        for nm in names:
            soft = t["sharp"] if nm == "sem_sharp" else t["soft"]
            save_table(p, nm, t["hard"], k, soft)
            save_table(p, nm + "_shuf", t["hard"][perm], k, soft[perm])
            record(nm, t["hard"], k)
    for e in cc.extra_embedders:
        t = cluster_tables(emb(e), owner, n, base_mask, cc.main_k, p)
        save_table(p, e, t["hard"], cc.main_k, t["soft"])
        record(e, t["hard"], cc.main_k)
    summary["domain"] = dict(k=len(meta["domains"]), nmi_domain=1.0)
    write_json(out / "summary.json", summary)
    mark_done(out, "cluster")
