"""Stage `cluster`: label-free paper clusters and every group table the experiments use.

1. Embed each paper's first ``embed_tokens`` tokens with a frozen encoder
   (e5-base-v2 with the "passage: " prefix, MiniLM, or tf-idf + SVD).
2. Fit spherical k-means with balanced (Sinkhorn) assignments on a sample of base-split papers.
3. Assign every paper with Sinkhorn over all papers: ``soft`` (temp soft_temp), ``sharp`` (sharp_temp),
   ``hard`` = argmax of sharp. Cluster sizes are therefore balanced.
4. Shuffled controls permute whole rows across papers, so every shuffled cluster has exactly the mass of
   its semantic twin but random content ("persistent, mass-matched random paper groupings", doc §2.5).

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

ENCODERS = {"e5": ("intfloat/e5-base-v2", "passage: ", 512),
            "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", 256)}
SHUFFLE_SEED = 999


# ------------------------------------------------------------------------------ texts

def paper_heads(p: Profile) -> list[str]:
    """Decoded first embed_tokens tokens of every paper, in global paper-id order."""
    meta, docs = load_meta(p), load_docs(p)
    tok = Tokenizer(meta["tokenizer"])
    texts = []
    for s_idx, s in enumerate(meta["split_order"]):
        stream = np.memmap(data_dir(p) / f"{s}.tokens.bin", dtype=np.uint16, mode="r")
        lens = docs["n_tokens"][docs["split"] == s_idx]
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for st, n in zip(starts, lens):
            texts.append(tok.decode(stream[st: st + min(n, p.cluster.embed_tokens)]))
    assert len(texts) == meta["n_docs"]
    return texts


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
    name, prefix, max_len = ENCODERS[embedder]
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


def cluster_tables(X: np.ndarray, base_mask: np.ndarray, k: int, p: Profile) -> dict:
    cc = p.cluster
    rng = np.random.default_rng(0)
    fit_idx = np.flatnonzero(base_mask)
    fit_idx = rng.choice(fit_idx, min(cc.fit_docs, len(fit_idx)), replace=False)
    C = fit_centroids(X[fit_idx], k, cc.kmeans_iters, cc.sinkhorn_iters)
    S = torch.from_numpy(X) @ C.T
    soft = sinkhorn(S, cc.soft_temp, cc.sinkhorn_iters).numpy()
    sharp = sinkhorn(S, cc.sharp_temp, cc.sinkhorn_iters).numpy()
    hard = sharp.argmax(1)
    return dict(hard=hard, soft=soft, sharp=sharp)


# ------------------------------------------------------------------------------ stage

def save_table(p: Profile, name: str, hard, k, soft=None) -> None:
    d = groups_dir(p)
    d.mkdir(parents=True, exist_ok=True)
    kw = dict(hard=np.asarray(hard, np.int16), k=np.int64(k))
    if soft is not None:
        kw["soft"] = np.asarray(soft, np.float16)
    np.savez(d / f"{name}.npz", **kw)


def build(p: Profile) -> None:
    out = groups_dir(p)
    if is_done(out, "cluster"):
        log(f"group tables already built in {out}")
        return
    meta, docs = load_meta(p), load_docs(p)
    n = meta["n_docs"]
    base_mask = docs["split"] == meta["split_order"].index("base")
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    summary = {}

    dom = docs["domain"].astype(np.int64)
    save_table(p, "domain", dom, len(meta["domains"]))
    save_table(p, "domain_shuf", dom[perm], len(meta["domains"]))

    log("reading paper heads")
    texts = paper_heads(p)
    emb_cache = {}

    def emb(name):
        if name not in emb_cache:
            log(f"embedding {n:,} papers with {name}")
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
        t = cluster_tables(emb(cc.embedder), base_mask, k, p)
        names = ["sem", "sem_sharp"] if k == cc.main_k else []
        if k in cc.alt_ks:
            names.append(f"sem_k{k}")
        for nm in names:
            soft = t["sharp"] if nm == "sem_sharp" else t["soft"]
            save_table(p, nm, t["hard"], k, soft)
            save_table(p, nm + "_shuf", t["hard"][perm], k, soft[perm])
            record(nm, t["hard"], k)
    for e in cc.extra_embedders:
        t = cluster_tables(emb(e), base_mask, cc.main_k, p)
        save_table(p, e, t["hard"], cc.main_k, t["soft"])
        record(e, t["hard"], cc.main_k)
    summary["domain"] = dict(k=len(meta["domains"]), nmi_domain=1.0)
    write_json(out / "summary.json", summary)
    mark_done(out, "cluster")
