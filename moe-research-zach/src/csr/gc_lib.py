"""csr-gradient-consistent-latent-edl-v1 (gc) library (docs/GC_EDL_V1_PREREG.md, sections 2-6).

  * opaque paper IDs, hash partitions, label firewall for discovery programs
  * paper views (>= 256-token separation), streamed CountSketch/TensorSketch of the per-view gradient of the scout's
    final MLP output projection (never materialising per-example gradients), exact Frobenius norm via Gram matrices
  * nuisance features, ridge, balanced soft prototypes (Sinkhorn swapped prediction + cross-view gradient coherence)
  * label-free discovery metrics (kappa, Hungarian-matched kappa)
  * GC-EDL loss = fsp_model.edl_loss with the one-hot label replaced by a per-paper soft assignment q
New module; nothing here reads domain labels."""
from __future__ import annotations

import hashlib
import json
import math
import re
import string
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .gc_paths import M_PROJ, PROJ_SEED  # noqa: F401

# --------------------------------------------------------------------------------------------- ids and partitions


def opaque_id(paper: str) -> str:
    return hashlib.sha256(f"gcedl-v1-opaque|{paper}".encode()).hexdigest()[:20]


def u_hash(salt: str, s: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{salt}|{s}".encode()).digest()[:8], "little") / 2**64


def disc_part(o: str) -> int:
    """0 = DISC-FIT, 1 = DISC-HOLD (by opaque-ID hash)."""
    return 0 if u_hash("gcedl-v1-disc", o) < 0.8 else 1


def fold_of(o: str, k=5) -> int:
    return int(u_hash("gcedl-v1-fold", o) * k)


# --------------------------------------------------------------------------------------------- label firewall
FIREWALL_PATTERNS = ("_docs.json", "forecast-specialist-pretraining-v2/data", "/labels/", "windows_", "/sealed/",
                     "/locked/", "m2d2")
_FIREWALL_ON = False


def firewall_blocks(path: str) -> bool:
    p = str(path)
    pl = p.lower()
    if any(s in pl for s in FIREWALL_PATTERNS):
        return True
    root = str(guard.ROOT)
    return pl.startswith(root.lower()) and "/.venv/" not in pl and "domain" in pl[len(root):]


def install_label_firewall():
    """Audit hook for discovery programs: any open() of a label-bearing path raises PermissionError."""
    global _FIREWALL_ON
    if _FIREWALL_ON:
        return

    def hook(event, args):
        if event == "open" and args:
            p = args[0]
            if isinstance(p, (str, bytes)) or hasattr(p, "__fspath__"):
                ps = p.decode() if isinstance(p, bytes) else str(p)
                if firewall_blocks(ps):
                    raise PermissionError(f"GC label firewall: access to {ps!r} is not permitted in discovery")
    sys.addaudithook(hook)
    _FIREWALL_ON = True


# --------------------------------------------------------------------------------------------- views


def view_starts(N: int, V: int = 256):
    """Starts of up to 4 non-overlapping V-token views separated by >= V tokens (prereg 3)."""
    if N < V:
        return []
    nv = min(4, (N + V) // (2 * V))
    if nv <= 1:
        return [(N - V) // 2]
    return [int(round(i * (N - V) / (nv - 1))) for i in range(nv)]


# --------------------------------------------------------------------------------------------- sketch


class Sketch:
    """CountSketch of vec(G), G (d_out, d_in) = sum_t delta_t a_t^T, with TensorSketch hash
    h(i, k) = (h2(i) + h1(k)) mod m and sign s2(i) s1(k). Fixed seed."""

    def __init__(self, d_in=768, d_out=256, m=128, seed=PROJ_SEED):
        rng = np.random.default_rng(seed)
        self.d_in, self.d_out, self.m, self.seed = d_in, d_out, m, seed
        self.h1 = rng.integers(0, m, d_in)
        self.s1 = rng.choice(np.array([-1.0, 1.0]), d_in)
        self.h2 = rng.integers(0, m, d_out)
        self.s2 = rng.choice(np.array([-1.0, 1.0]), d_out)
        self._mats = {}

    def spec(self):
        arr = np.concatenate([self.h1, self.s1, self.h2, self.s2]).astype(np.float64)
        return {"type": "CountSketch/TensorSketch", "d_in": self.d_in, "d_out": self.d_out, "m": self.m,
                "seed": self.seed, "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}

    def mats(self, device, dtype=torch.float64):
        key = (str(device), dtype)
        if key not in self._mats:
            S1 = torch.zeros(self.d_in, self.m, dtype=dtype)
            S1[torch.arange(self.d_in), torch.as_tensor(self.h1)] = torch.as_tensor(self.s1, dtype=dtype)
            S2 = torch.zeros(self.d_out, self.m, dtype=dtype)
            S2[torch.arange(self.d_out), torch.as_tensor(self.h2)] = torch.as_tensor(self.s2, dtype=dtype)
            self._mats[key] = (S1.to(device), S2.to(device))
        return self._mats[key]

    def explicit(self):
        """Explicit P (m, d_out * d_in) for row-major vec(G) (tests only)."""
        P = np.zeros((self.m, self.d_out * self.d_in))
        for i in range(self.d_out):
            for k in range(self.d_in):
                P[(self.h2[i] + self.h1[k]) % self.m, i * self.d_in + k] = self.s2[i] * self.s1[k]
        return P

    def apply(self, a, d):
        """a (B,T,d_in), d (B,T,d_out) -> (B, m) sketch of sum_t d_t a_t^T (float64)."""
        S1, S2 = self.mats(a.device)
        c1 = a.to(torch.float64) @ S1
        c2 = d.to(torch.float64) @ S2
        Fq = (torch.fft.rfft(c1, dim=-1) * torch.fft.rfft(c2, dim=-1)).sum(1)
        return torch.fft.irfft(Fq, n=self.m, dim=-1)


def gram_norm(a, d):
    """Exact ||sum_t d_t a_t^T||_F per batch row via Gram matrices."""
    a64, d64 = a.to(torch.float64), d.to(torch.float64)
    ga = a64 @ a64.transpose(1, 2)
    gd = d64 @ d64.transpose(1, 2)
    return (ga * gd).sum((1, 2)).clamp(min=0).sqrt()


def scout_parts(scout, ids):
    """Forward the frozen scout on views (B,V); returns (a, xo, cos/sin/mask unused) where a = W2 input of the final
    block and xo = final residual stream (requires grad) for the delta computation."""
    from .model import CSRModel, rope_cos_sin
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).expand(B, T)
    doc = torch.zeros_like(pos)
    with torch.no_grad():
        x = scout.tok_emb(ids)
        cos, sin = rope_cos_sin(pos, scout.cfg.d_model // scout.cfg.n_head, scout.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = CSRModel.attention_mask(doc, pos)
        for blk in scout.blocks[:-1]:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            x = x + blk.ffn(blk.norm2(x))
        blk = scout.blocks[-1]
        x = x + blk.attn(blk.norm1(x), cos, sin, mask)
        h = blk.norm2(x)
        a = F.silu(blk.ffn.w1(h)) * blk.ffn.w3(h)
        y = blk.ffn.w2(a)
    xo = (x + y).detach().requires_grad_(True)
    return a, xo


def scout_fingerprints(scout, ids, sk: Sketch):
    """Per-view: g = P vec(grad_{W2} mean_token_CE) (B, m) float64, exact ||grad||_F (B,), mean CE (B,).
    Only the view's own tokens are inputs (positions 0..V-1)."""
    a, xo = scout_parts(scout, ids)
    hf = scout.norm_f(xo)
    logits = hf[:, :-1] @ scout.tok_emb.weight.t()
    lp = torch.log_softmax(logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits, -1)
    ce_t = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    ce = ce_t.mean(1)
    (delta,) = torch.autograd.grad(ce.sum(), xo)
    g = sk.apply(a, delta)
    nrm = gram_norm(a, delta)
    return g.detach(), nrm.detach(), ce.detach().double()


# --------------------------------------------------------------------------------------------- encoders


class BGE:
    """Frozen BAAI/bge-small-en-v1.5 (CLS pooling, L2 norm). Independent of MiniLM/drafts, the scout and e5."""

    def __init__(self, device, path):
        from transformers import BertModel, BertTokenizerFast
        self.sha = guard.sha256_file(path / "model.safetensors")
        self.tk = BertTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), vocab_file=str(path / "vocab.txt"))
        self.model = BertModel.from_pretrained(str(path), local_files_only=True, add_pooling_layer=False,
                                               use_safetensors=True).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts, max_len=512, bs=512):
        out = np.zeros((len(texts), self.model.config.hidden_size), np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tk([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                          return_tensors="pt").to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                h = self.model(**enc).last_hidden_state[:, 0].float()
            out[idx] = F.normalize(h, dim=-1).cpu().numpy()
        return out


# --------------------------------------------------------------------------------------------- nuisance features
_CITE = re.compile(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]|\bet al\b|\\cite")
_PUNCT = set(string.punctuation)
_MATH = set("\\$^_{}=<>|~")
NUIS_NAMES = (["scout_ce", "log_grad_norm", "log_paper_len", "rel_pos", "log_nseg", "mean_logp_unigram",
               "frac_top100", "type_token_ratio"] + [f"hash{i}" for i in range(16)] +
              ["punct_rate", "digit_rate", "math_markup_rate", "upper_rate", "whitespace_rate", "citation_per_kchar",
               "mean_word_len"])


def text_features(t: str):
    n = max(1, len(t))
    letters = sum(c.isalpha() for c in t)
    words = t.split()
    return [sum(c in _PUNCT for c in t) / n, sum(c.isdigit() for c in t) / n, sum(c in _MATH for c in t) / n,
            sum(c.isupper() for c in t) / max(1, letters), sum(c.isspace() for c in t) / n,
            1000.0 * len(_CITE.findall(t)) / n, (sum(len(w) for w in words) / len(words)) if words else 0.0]


def token_features(tok, logp_uni, top_mask):
    """tok (n, V) int -> (n, 1+1+1+16) [mean logp, frac top100, ttr, 16 hash buckets]."""
    tok = np.asarray(tok, np.int64)
    n, V = tok.shape
    lp = logp_uni[tok].mean(1)
    tp = top_mask[tok].mean(1)
    ttr = np.asarray([len(np.unique(r)) / V for r in tok])
    hb = ((tok * 2654435761) % 2**32) >> 28
    hh = np.stack([(hb == b).mean(1) for b in range(16)], 1)
    return np.concatenate([lp[:, None], tp[:, None], ttr[:, None], hh], 1)


# --------------------------------------------------------------------------------------------- ridge


def ridge_fit(X, Y, alpha):
    """X (n,p) standardized, Y (n,q). Returns (W (p,q), b (q,))."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, ym - xm @ W


def ridge_pred(Wb, X):
    W, b = Wb
    return X @ W + b


def r2(Y, Yh, mean=None):
    mu = Y.mean(0) if mean is None else mean
    return 1.0 - ((Y - Yh) ** 2).sum(0) / np.maximum(1e-12, ((Y - mu) ** 2).sum(0))


def grouped_cv_alpha(X, Y, folds, alphas):
    best, scores = None, {}
    for al in alphas:
        sse, sst = 0.0, 0.0
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            Wb = ridge_fit(X[tr], Y[tr], al)
            sse += ((Y[te] - ridge_pred(Wb, X[te])) ** 2).sum()
            sst += ((Y[te] - Y[tr].mean(0)) ** 2).sum()
        scores[al] = 1.0 - sse / sst
        if best is None or scores[al] > scores[best]:
            best = al
    return best, scores


# --------------------------------------------------------------------------------------------- prototypes


def sinkhorn(scores, eps=0.05, iters=3):
    """SwAV balanced assignment over the batch. scores (B,K) -> (B,K) rows sum to 1."""
    Q = torch.exp((scores - scores.max()) / eps).t()  # (K,B)
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True)
        Q /= K
        Q /= Q.sum(0, keepdim=True)
        Q /= B
    return (Q * B).t()


class ProtoNet(torch.nn.Module):
    def __init__(self, d_in, K, dz=64, tau=0.1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + seed)
        self.W = torch.nn.Linear(d_in, dz)
        with torch.no_grad():
            self.W.weight.copy_(torch.randn(dz, d_in, generator=g) / math.sqrt(d_in))
            self.W.bias.zero_()
        self.C = torch.nn.Parameter(torch.randn(K, dz, generator=g))
        self.tau = tau

    def scores(self, x):
        z = F.normalize(self.W(x), dim=-1)
        return z @ F.normalize(self.C, dim=-1).t()

    def forward(self, x):
        return self.scores(x) / self.tau


def fit_prototypes(X, nv, Fh, K, lam_coh, seed, epochs=30, batch=2048, lr=1e-3, tau=0.1, lam_bal=1.0, lam_ent=0.05,
                   device="cpu", log=None):
    """X (P,4,d) float32 views (padded), nv (P,) >= 2, Fh (P,4,m) unit functional fingerprints. Returns (net, hist)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 + seed)
    P, _, d = X.shape
    Xt = torch.as_tensor(X, device=device)
    Ft = torch.as_tensor(Fh, device=device)
    net = ProtoNet(d, K, tau=tau, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mu = None
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(P)
        for s in range(0, P - batch // 2, batch):
            b = perm[s:s + batch]
            n_b = nv[b]
            i = rng.integers(0, n_b)
            j = (i + 1 + rng.integers(0, n_b - 1)) % n_b
            bt = torch.as_tensor(b, device=device)
            it, jt = torch.as_tensor(i, device=device), torch.as_tensor(j, device=device)
            x1, x2 = Xt[bt, it], Xt[bt, jt]
            f1, f2 = Ft[bt, it], Ft[bt, jt]
            s1, s2 = net.scores(x1), net.scores(x2)
            l1, l2 = s1 / tau, s2 / tau
            with torch.no_grad():
                t1, t2 = sinkhorn(s1.detach()), sinkhorn(s2.detach())
            lp1, lp2 = torch.log_softmax(l1, -1), torch.log_softmax(l2, -1)
            cons = -0.5 * ((t2 * lp1).sum(1).mean() + (t1 * lp2).sum(1).mean())
            p1, p2 = lp1.exp(), lp2.exp()
            if mu is None:
                mu = F.normalize(torch.randn(K, Ft.shape[-1], generator=torch.Generator().manual_seed(seed))
                                 .to(device, Ft.dtype), dim=-1)
            coh = torch.zeros((), device=device)
            if lam_coh > 0:
                g1 = F.normalize(p1 @ mu.to(p1.dtype), dim=-1)
                g2 = F.normalize(p2 @ mu.to(p2.dtype), dim=-1)
                coh = -0.5 * (F.cosine_similarity(g1, f2.to(g1.dtype), -1).mean() +
                              F.cosine_similarity(g2, f1.to(g2.dtype), -1).mean())
            pbar = 0.5 * (p1.mean(0) + p2.mean(0))
            bal = (pbar * torch.log(pbar.clamp(min=1e-12) * K)).sum()
            ent = 0.5 * ((-(p1 * lp1).sum(1)).mean() + (-(p2 * lp2).sum(1)).mean()) / math.log(K)
            loss = cons + lam_coh * coh + lam_bal * bal + lam_ent * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # lagged centroid update (after the loss)
                cs = t1.t().to(f1.dtype) @ f1 + t2.t().to(f2.dtype) @ f2
                mu = F.normalize(0.9 * mu + 0.1 * F.normalize(cs, dim=-1), dim=-1)
        hist.append({"epoch": ep, "loss": float(loss), "cons": float(cons), "coh": float(coh), "bal": float(bal),
                     "ent": float(ent)})
        if log:
            log(hist[-1])
    net.mu = mu
    return net, hist


@torch.no_grad()
def view_logits(net, X, device="cpu", bs=65536):
    flat = torch.as_tensor(X.reshape(-1, X.shape[-1]))
    out = torch.cat([net(flat[s:s + bs].to(device)).cpu() for s in range(0, len(flat), bs)])
    return out.reshape(X.shape[0], X.shape[1], -1).numpy().astype(np.float64)


def paper_q(vl, nv):
    """q[d] = softmax(mean over the paper's views of the assignment logits); zeros for papers without views."""
    P, V, K = vl.shape
    m = (np.arange(V)[None, :] < nv[:, None]).astype(np.float64)
    ml = (vl * m[..., None]).sum(1) / np.maximum(1, nv)[:, None]
    q = np.exp(ml - ml.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    q[nv == 0] = 0.0
    return q


# --------------------------------------------------------------------------------------------- label-free metrics


def cohen_kappa(a, b, K):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    po = float((a == b).mean())
    pa = np.bincount(a, minlength=K) / len(a)
    pb = np.bincount(b, minlength=K) / len(b)
    pe = float((pa * pb).sum())
    return (po - pe) / max(1e-12, 1 - pe)


def matched_kappa(a, b, K):
    """Hungarian-match labels of b to a (max agreement), then Cohen's kappa."""
    from .lfr_eval import hungarian
    C = np.zeros((K, K))
    np.add.at(C, (np.asarray(a, np.int64), np.asarray(b, np.int64)), 1.0)
    col = hungarian(-C)  # row r of a matched to column col[r] of b
    inv = np.empty(K, np.int64)
    inv[col] = np.arange(K)
    return cohen_kappa(a, inv[np.asarray(b, np.int64)], K), inv


# --------------------------------------------------------------------------------------------- GC-EDL loss


def gc_edl_loss(p, qseq, seq_id, nonpad, eps=1e-8, min_mass=1e-6):
    """fsp_model.edl_loss with soft group weights. p (N,E) token router softmax; qseq (S,K) per-sequence soft
    assignment (zero rows = no assignment); seq_id (N,); nonpad (N,). Per-sequence mean -> q-weighted group means ->
    mean over group pairs of -log(JSD + eps). Groups with total weight < min_mass in this batch are skipped."""
    m = nonpad & (seq_id >= 0)
    p, sid = p[m], seq_id[m]
    if p.shape[0] == 0:
        return p.new_zeros(())
    S = qseq.shape[0]
    E = p.shape[1]
    ssum = p.new_zeros(S, E).index_add_(0, sid, p)
    scnt = p.new_zeros(S).index_add_(0, sid, torch.ones_like(sid, dtype=p.dtype))
    ok = scnt > 0
    smean = ssum[ok] / scnt[ok].unsqueeze(1)
    q = qseq[ok].to(p.dtype)
    mass = q.sum(0)
    keep = mass > min_mass
    if int(keep.sum()) < 2:
        return p.new_zeros(())
    qk = q[:, keep]
    P = (qk.t() @ smean) / mass[keep].unsqueeze(1)
    Mg = P.shape[0]
    a, b = torch.triu_indices(Mg, Mg, 1, device=p.device)
    pa, pb = P[a], P[b]
    mm = 0.5 * (pa + pb)
    kl = lambda x, y: (x * (torch.log(x.clamp(min=1e-12)) - torch.log(y.clamp(min=1e-12)))).sum(-1)  # noqa: E731
    jsd = (0.5 * kl(pa, mm) + 0.5 * kl(pb, mm)).clamp(min=0.0)
    return (-torch.log(jsd + eps)).mean()


def mean_pair_jsd(p, qseq, seq_id, nonpad):
    """Diagnostic: mean pairwise JSD of the q-weighted group routing means (no gradient)."""
    with torch.no_grad():
        val = gc_edl_loss(p, qseq, seq_id, nonpad)
        return float(torch.exp(-val)) if val.numel() else 0.0


class PaperQ:
    """Frozen per-paper assignment table (npz: opaque (P,), q (P,K)) mapped onto a split's documents."""

    def __init__(self, path, split, expect_sha=None):
        self.sha = guard.sha256_file(path)
        if expect_sha is not None and self.sha != expect_sha:
            raise guard.GuardError(f"assignment table hash mismatch {path}")
        z = np.load(path)
        idx = {str(o): i for i, o in enumerate(z["opaque"])}
        q = z["q"].astype(np.float32)
        self.K = q.shape[1]
        cache = {}
        self.q_of_doc = np.zeros((split.n_docs, self.K), np.float32)
        hit = 0
        for di, p in enumerate(split.papers):
            o = cache.get(p)
            if o is None:
                o = cache[p] = opaque_id(p)
            i = idx.get(o)
            if i is not None:
                self.q_of_doc[di] = q[i]
                hit += 1
        self.coverage = hit / max(1, split.n_docs)

    def qseq(self, npb):
        """Per-sequence q (S,K) for a make_batch dict (seq ids 0..S-1)."""
        sid = npb["seq_id"].reshape(-1)
        gd = npb["gdoc"].reshape(-1)
        S = int(sid.max()) + 1 if (sid >= 0).any() else 0
        seq_g = np.full(S, -1, np.int64)
        ok = sid >= 0
        seq_g[sid[ok]] = gd[ok]
        return self.q_of_doc[seq_g]


def jdump(obj):
    return json.dumps(obj, sort_keys=True, default=str)
