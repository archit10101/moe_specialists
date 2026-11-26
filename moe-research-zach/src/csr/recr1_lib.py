"""recr1 core: data loading over retained spb1 tables, causal candidate-head features, the candidate-proposal head,
the exact vectorised RECR policy simulator, controls, compute accounting and metrics.

Conventions. d[t, e] = CE_e(t) - CE_base(t) (nats; negative = the residual helps). Realised residual evidence of an
executed residual e at t is g = -d[t, e]; it becomes observable only after x_{t+1} is revealed, i.e. it may affect
routing at positions >= t+1 only (enforced by the simulator's update order and tested in recr1_tests.py).
Options: 0..31 = individual specialists of the bank, OPT_S = 32 (the existing unrestricted token router's choice),
OPT_G = 33 (generic path = GEN32 bank under R_GEN), OPT_BASE = 34 (no residual)."""
import json
import math
import time

import numpy as np
import torch

from csr import guard
from csr import recr1_paths as P

OPT_S, OPT_G, OPT_BASE = 32, 33, 34
N_OPT = 34                      # head outputs: 32 specialists + S + G
TWIN = {"SEM2": "R_SEM", "SHUF2": "R_SHUF"}


# ----------------------------------------------------------------------------------------------------------- data
def partition_papers(name):
    dm = json.loads((P.ART / "data_manifest.json").read_text())
    if P.SMOKE:
        return np.array(dm["partitions"]["UTRAIN"]["paper_index"][:P.SMOKE_PAPERS] if name == "UTRAIN" else
                        dm["partitions"]["UTRAIN"]["paper_index"][P.SMOKE_PAPERS:2 * P.SMOKE_PAPERS], dtype=np.int64)
    return np.array(dm["partitions"][name]["paper_index"], dtype=np.int64)


class Part:
    """All per-token arrays of one bank seed restricted to one partition (whole documents, original order)."""

    def __init__(self, seed, name, banks=("SEM2", "SHUF2")):
        c = np.load(P.EXTRACT / f"common_s{seed}/common_SEL.npz")
        keep = np.isin(c["paper"].astype(np.int64), partition_papers(name))
        self.idx = np.nonzero(keep)[0]
        self.name, self.seed = name, seed
        g = lambda k: c[k][self.idx]
        self.pos = g("pos").astype(np.int64)
        self.doc_raw = g("doc").astype(np.int64)
        self.paper_raw = g("paper").astype(np.int64)
        self.lab_sem = g("lab_sem").astype(np.int64)
        self.lab_shuf = g("lab_shuf").astype(np.int64)
        self.r = g("lab_rsem").astype(np.int64)
        self.rp = g("lab_rsem_prev").astype(np.int64)
        self.tgt = g("tgt").astype(np.int64)
        self.ce_base = g("ce_base").astype(np.float64)
        self.doclen = g("doclen").astype(np.int64)
        self.N = len(self.idx)
        starts = np.nonzero(self.pos == 0)[0]
        assert len(starts) and starts[0] == 0 and np.all(np.diff(self.doc_raw)[np.diff(self.doc_raw) != 0] != 0)
        self.starts = starts
        self.lens = np.diff(np.r_[starts, self.N])
        assert np.array_equal(self.pos, np.concatenate([np.arange(n) for n in self.lens]))
        self.n_docs = len(starts)
        self.Tm = int(self.lens.max())
        self.doc = np.repeat(np.arange(self.n_docs), self.lens)
        up, self.paper = np.unique(self.paper_raw, return_inverse=True)
        self.P = len(up)
        self.tables = {}
        for b in banks + ("GEN32",):
            z = np.load(P.EXTRACT / f"g0_s{seed}_S1/bank_{b}_SEL.npz")
            self.tables[b] = z["d"][self.idx].astype(np.float32)
            if b == "SEM2":
                self.e_tr = z["e_tr"][self.idx].astype(np.int64)
        self.routes = {}

    def route(self, tag, r):
        k = (tag, r)
        if k not in self.routes:
            z = np.load(P.ROUTES / f"{tag}_s{self.seed}_r{r}_SEL.npz")
            self.routes[k] = {f: z[f][self.idx] for f in z.files}
        return self.routes[k]

    def path_d(self, bank, r):
        """(d_S, e_S, d_G): exact per-token CE delta of the bank's existing unrestricted router and of the generic path."""
        rs = self.route(TWIN[bank], r)
        eS = rs["e"].astype(np.int64)
        dS = self.tables[bank][np.arange(self.N), eS]
        eG = self.route("R_GEN", r)["e"].astype(np.int64)
        dG = self.tables["GEN32"][np.arange(self.N), eG]
        return dS, eS, dG

    def masks(self):
        m = {"all": np.ones(self.N, bool), "pos_0_31": self.pos < 32, "pos_0_7": self.pos < 8,
             "pos_32_127": (self.pos >= 32) & (self.pos < 128), "pos_ge128": self.pos >= 128}
        ch = (self.rp >= 0) & (self.r != self.rp)
        m["transition_first8"] = ch & (self.pos % 256 < 8)
        m["transition_first32"] = ch & (self.pos % 256 < 32)
        m["stable_context"] = (self.rp >= 0) & (self.r == self.rp)
        return m

    def padded(self, x, fill=0):
        """(N, ...) -> (n_docs, Tm, ...) padded."""
        out = np.full((self.n_docs, self.Tm) + x.shape[1:], fill, dtype=x.dtype)
        out[self.doc, self.pos] = x
        return out


# ------------------------------------------------------------------------------------------------------- features
def causal_hist(part, e, n, halflife=None):
    """Per-token causal histogram of categorical router outputs e (values 0..n-1) within each document, INCLUDING the
    current token (router outputs at t are functions of h_<=t). halflife None = prefix mean."""
    Ep = part.padded(e.astype(np.int64), fill=-1)
    D, T = Ep.shape
    H = np.zeros((D, T, n), np.float32)
    acc = np.zeros((D, n), np.float64)
    a = 0.0 if halflife is None else 0.5 ** (1.0 / halflife)
    rows = np.arange(D)
    for t in range(T):
        valid = Ep[:, t] >= 0
        if halflife is None:
            acc[rows[valid], Ep[valid, t]] += 1.0
            H[:, t] = acc / (t + 1)
        else:
            acc *= a
            acc[rows[valid], Ep[valid, t]] += (1 - a)
            H[:, t] = acc / (1 - a ** (t + 1))
    return H[part.doc, part.pos]


def features(part, bank, r):
    """Causal candidate-head features (198 dims), only from retained causal router outputs at positions <= t:
    S-router (R_SEM / R_SHUF twin) choice/confidence, R_COARSE16 choice/confidence, R_GEN confidence, position."""
    rs, rc, rg = part.route(TWIN[bank], r), part.route("R_COARSE16", r), part.route("R_GEN", r)
    eS, eC = rs["e"].astype(np.int64), rc["e"].astype(np.int64)
    f = [np.eye(32, dtype=np.float32)[eS], causal_hist(part, eS, 32, 16), causal_hist(part, eS, 32, 128),
         causal_hist(part, eS, 32), np.eye(16, dtype=np.float32)[eC], causal_hist(part, eC, 16, 16),
         causal_hist(part, eC, 16, 128), causal_hist(part, eC, 16),
         np.stack([rs["p1"].astype(np.float32), rs["marg"].astype(np.float32) / 8, rg["p1"].astype(np.float32),
                   rc["p1"].astype(np.float32), np.log1p(part.pos).astype(np.float32) / 7,
                   np.ones(part.N, np.float32)], 1)]
    return np.concatenate(f, 1)


def targets(part, bank, r):
    """Head targets: realised gain vs base (-d) of the 32 specialists, the S path and the G path (frozen tables)."""
    dS, _, dG = part.path_d(bank, r)
    return -np.concatenate([part.tables[bank], dS[:, None], dG[:, None]], 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------- head
class Head(torch.nn.Module):
    def __init__(self, n_in, kind, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        if kind == "linear":
            self.net = torch.nn.Linear(n_in, N_OPT)
            dims = [(n_in, N_OPT)]
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.GELU(), torch.nn.Linear(64, N_OPT))
            dims = [(n_in, 64), (64, N_OPT)]
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) / math.sqrt(m.weight.shape[1]) * 0.1)
                    m.bias.zero_()
        self.macs = sum(a * b for a, b in dims)

    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_head(X, Y, kind, seed, tokens, ckpt_path, log=print, resume=True, stop_check=None):
    """Huber regression of 34 utility targets; deterministic (CPU, fixed generator); atomic checkpoints every
    HEAD_CKPT_TOKENS tokens or HEAD_CKPT_SECONDS seconds, each re-read for verification. Returns the trained head."""
    torch.manual_seed(seed)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y * 10.0)       # targets in 0.1-nat units
    mu, sd = Xt.mean(0), Xt.std(0).clamp(min=1e-3)
    sd[-1] = 1.0
    mu[-1] = 0.0
    head = Head(X.shape[1], kind, seed)
    opt = torch.optim.AdamW(head.parameters(), lr=P.HEAD_LR, weight_decay=P.HEAD_WD)
    steps = tokens // P.HEAD_BATCH
    step0 = 0
    if resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
    gen = torch.Generator().manual_seed(seed + 99)
    for _ in range(step0):                       # replay the sampler so resume is bitwise
        torch.randint(0, len(Xt), (P.HEAD_BATCH,), generator=gen)
    last_t, last_tok = time.time(), step0 * P.HEAD_BATCH
    for step in range(step0, steps):
        lr = P.HEAD_LR * 0.5 * (1 + math.cos(math.pi * step / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        b = torch.randint(0, len(Xt), (P.HEAD_BATCH,), generator=gen)
        out = head((Xt[b] - mu) / sd)
        loss = torch.nn.functional.huber_loss(out, Yt[b], delta=P.HUBER * 10)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tok = (step + 1) * P.HEAD_BATCH
        if (tok - last_tok >= P.HEAD_CKPT_TOKENS or time.time() - last_t >= P.HEAD_CKPT_SECONDS or step + 1 == steps):
            save_head(ckpt_path, head, opt, step + 1, mu, sd, kind, seed)
            last_t, last_tok = time.time(), tok
            log({"step": step + 1, "tokens": tok, "loss": float(loss.detach())})
        if stop_check is not None and stop_check():
            save_head(ckpt_path, head, opt, step + 1, mu, sd, kind, seed)
            raise KeyboardInterrupt("stop requested")
    head.mu, head.sd = mu, sd
    return head


def save_head(path, head, opt, step, mu, sd, kind, seed):
    obj = {"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "mu": mu, "sd": sd, "kind": kind,
           "seed": seed}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    chk = torch.load(tmp, map_location="cpu", weights_only=False)          # readability check before rename
    assert chk["step"] == step
    tmp.replace(path)


def load_head(path, n_in):
    st = torch.load(path, map_location="cpu", weights_only=False)
    h = Head(n_in, st["kind"], st["seed"])
    h.load_state_dict(st["head"])
    h.mu, h.sd = st["mu"], st["sd"]
    return h


@torch.no_grad()
def predict(head, X):
    out = []
    for i in range(0, len(X), 262144):
        x = torch.from_numpy(X[i:i + 262144])
        out.append((head((x - head.mu) / head.sd) / 10.0).numpy())
    return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------------------------------------------ simulator
def calibrate_kappa(part, dS, gamma, tau, rate, stride, norm="raw", norm_sd=None):
    """Label-free threshold: kappa such that the CUSUM on the S path's realised evidence (reset after each alarm)
    alarms `rate` times per 1000 tokens on the given (UTRAIN) partition. Bisection on kappa."""
    g = -dS.astype(np.float64)
    if norm == "z":
        g = g / norm_sd
    elif norm == "clip":
        g = np.clip(g, -0.2, 0.2)
    G = part.padded(g, fill=np.nan)

    def alarms(kappa):
        D, T = G.shape
        C = np.zeros(D)
        n = 0
        for t in range(T):
            x = G[:, t]
            v = ~np.isnan(x) & ((t % stride) == 0)
            C = np.where(v, np.maximum(0.0, gamma * C + tau - np.nan_to_num(x)), C)
            a = v & (C > kappa)
            n += int(a.sum())
            C[a] = 0.0
        return 1000.0 * n / part.N

    lo, hi = 1e-4, 50.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if alarms(mid) > rate:
            lo = mid
        else:
            hi = mid
    return hi, alarms(hi)


def simulate(part, dB, dS, dG, U, cfg, kappa, mode="E", stride=1, norm_sd=None, oracle=None, collect=False):
    """Exact vectorised RECR simulation over all documents of `part`.

    dB (N,32) bank table; dS, dG (N,) S-path and generic-path deltas; U (N,34) causal head predictions (gain vs base;
    columns 0..31 specialists, 32 = S, 33 = G). mode 'E' = residual-evidence change detection + top-2 temporary search;
    'D' = change detection, fallback for B tokens, commit to the head's best mean predicted advantage (no shadows).
    oracle (diagnostics only, never deployable): dict with optional keys 'cand' (N,2) forced candidates,
    'commit_true' (use the realised hold-window mean instead of search evidence), 'alarm_at' (N,) bool forced alarms.

    Order at each position t (vectorised over documents): (1) decide the executed option from state built from
    positions < t and the head at t; (2) reveal d at t and update evidence/state for position t+1. Returns per-token
    executed option (-1 = base), its exact delta, shadows executed, evidence-computed flag, alarm flag, search flag."""
    B, nc = int(cfg["B"]), int(cfg["n_cand"])
    gamma, tau, theta = float(cfg["gamma"]), float(cfg["tau"]), float(cfg["abstain"])
    fb = OPT_G if cfg["fallback"] == "generic" else OPT_BASE
    probe = int(cfg.get("probe", 64))
    Dp = part.padded
    # all options' deltas: 0..31 bank, 32 S, 33 G, 34 base(0)
    Dall = np.concatenate([dB, dS[:, None], dG[:, None], np.zeros((part.N, 1), np.float32)], 1)
    DA = Dp(Dall.astype(np.float32), fill=0)
    adv = U[:, :OPT_G + 1] - U[:, OPT_G:OPT_G + 1]            # predicted advantage over G (G column = 0)
    AD = Dp(adv[:, :OPT_S + 1].astype(np.float32), fill=-1e9)
    norm = cfg.get("norm", "raw")
    sd_opt = None if norm_sd is None else np.asarray(norm_sd, np.float64)      # (35,) per option (z-normalisation)
    Dn, T = AD.shape[:2]
    rows = np.arange(Dn)
    lens = part.lens
    hold = int(cfg.get("hold_eval", 64))
    # state
    searching = np.ones(Dn, bool)
    s_count = np.zeros(Dn, np.int64)
    ev_sum = np.zeros((Dn, OPT_S + 1))
    ev_n = np.zeros((Dn, OPT_S + 1))
    pa_sum = np.zeros((Dn, OPT_S + 1))
    active = np.full(Dn, fb, np.int64)
    C = np.zeros(Dn)
    since = np.zeros(Dn, np.int64)
    out_opt = np.full((Dn, T), -9, np.int16)
    out_d = np.zeros((Dn, T), np.float32)
    out_sh = np.zeros((Dn, T), np.int8)
    out_ev = np.zeros((Dn, T), bool)
    out_alarm = np.zeros((Dn, T), bool)
    out_search = np.zeros((Dn, T), bool)
    commits = []
    for t in range(T):
        live = t < lens
        # ---------------- (1) decision for position t
        cand = np.full((Dn, nc), -1, np.int64)
        if mode == "E":
            pa_sum[searching] += AD[searching, t]
            if oracle is not None and "cand" in oracle:
                cand = Dp(oracle["cand"], fill=0)[:, t].astype(np.int64)
            else:
                cand = np.argsort(-pa_sum, 1)[:, :nc]
        elif mode == "D":
            pa_sum[searching] += AD[searching, t]
        ex = np.where(searching, fb, active)
        out_opt[:, t] = np.where(ex == OPT_BASE, -1, ex)
        out_d[:, t] = DA[rows, t, ex]
        out_search[:, t] = searching
        # ---------------- (2) reveal token t, update evidence/state for t+1
        if mode == "E":
            sh = searching[:, None] & live[:, None]
            dfb = DA[rows, t, fb]
            for j in range(nc):
                cj = cand[:, j]
                delta = dfb - DA[rows, t, cj]               # g_cand - g_fallback (both executed during search)
                m = sh[:, 0]
                ev_sum[rows[m], cj[m]] += delta[m]
                ev_n[rows[m], cj[m]] += 1
            out_sh[:, t] = np.where(searching & live, nc, 0)
        s_count[searching] += 1
        done = searching & (s_count >= B)
        if done.any():
            if mode == "E":
                if oracle is not None and oracle.get("commit_true"):
                    tt = min(T, t + 1 + hold)
                    fut = DA[:, t + 1:tt, :OPT_S + 1].mean(1) if tt > t + 1 else np.zeros((Dn, OPT_S + 1))
                    sc = np.where(ev_n >= max(1, B // 2), (DA[:, t + 1:tt, fb].mean(1)[:, None] if tt > t + 1 else 0) - fut, -1e9)
                else:
                    sc = np.where(ev_n >= max(1, B // 2), ev_sum / np.maximum(ev_n, 1), -1e9)
            else:
                sc = pa_sum / np.maximum(s_count[:, None], 1)
            best = sc.argmax(1)
            bs = sc[rows, best]
            new = np.where(bs > theta, best, fb)
            if collect:
                for i in np.nonzero(done & live)[0]:
                    top = np.argsort(-pa_sum[i])[:nc]
                    commits.append((int(i), int(t), int(new[i]), [int(x) for x in top],
                                    [float(sc[i, x]) if sc[i, x] > -1e8 else None for x in top]))
            active = np.where(done, new, active)
            searching = searching & ~done
            C[done] = 0.0
            since[done] = 0
            ev_sum[done] = 0
            ev_n[done] = 0
            pa_sum[done] = 0
            s_count[done] = 0
        stab = ~searching & ~done & live
        evm = stab & ((t % stride) == 0) & (active != OPT_BASE)
        g = -DA[rows, t, active].astype(np.float64)
        if norm == "z" and sd_opt is not None:
            g = g / sd_opt[active]
        elif norm == "clip":
            g = np.clip(g, -0.2, 0.2)
        C = np.where(evm, np.maximum(0.0, gamma * C + tau - g), C)
        out_ev[:, t] = evm
        alarm = evm & (C > kappa)
        since[stab] += 1
        alarm |= stab & (active == OPT_BASE) & (since >= probe)          # base-fallback probe (no evidence there)
        if oracle is not None and "alarm_at" in oracle:
            fa = Dp(oracle["alarm_at"], fill=False)[:, t]
            alarm = (stab & fa)
        out_alarm[:, t] = alarm
        if alarm.any():
            searching = searching | alarm
            s_count[alarm] = 0
            ev_sum[alarm] = 0
            ev_n[alarm] = 0
            pa_sum[alarm] = 0
            C[alarm] = 0.0
        # document boundary: next position of a finished doc is padding; new docs start searching at t=0 by init
    f = lambda a: a[part.doc, part.pos]
    res = {"opt": f(out_opt).astype(np.int16), "d": f(out_d), "shadows": f(out_sh), "ev": f(out_ev),
           "alarm": f(out_alarm), "search": f(out_search)}
    if collect:
        res["commits"] = commits
    return res


# ---------------------------------------------------------------------------------------------- compute accounting
def macs():
    a = P.ARCH
    d = a["d"]
    attn_proj = (3 * d * d + d * d) * a["L"]
    attn_ctx = 2 * (a["T"] // 2) * d * a["L"]
    ffn = 2 * a["dense_ffn"] * d + (a["L"] - 1) * (2 * a["shared"] * d + a["topk"] * 2 * a["expert_ffn"] * d + a["n_exp"] * d)
    return {"trunk": attn_proj + attn_ctx + ffn, "lm": d * a["vocab"], "res": 2 * d * a["res_ffn"],
            "router": a["router_params"]}


def policy_macs(sim, head_macs, n_routers):
    """Mean MACs/token of a simulated policy vs the existing one-residual S path (trunk + LM + 1 residual + 1 router)."""
    m = macs()
    ref = m["trunk"] + m["lm"] + m["res"] + m["router"]
    n_res = (sim["opt"] >= 0).astype(np.float64) + sim["shadows"]
    lm = 1.0 + sim["ev"].astype(np.float64) + sim["shadows"]
    cost = m["trunk"] + lm * m["lm"] + n_res * m["res"] + n_routers * m["router"] + head_macs
    return {"macs_per_token": float(cost.mean()), "ref_macs": float(ref), "overhead": float(cost.mean() / ref - 1),
            "residuals_per_token": float(n_res.mean()), "extra_lm_per_token": float((lm - 1).mean()),
            "search_frac": float(sim["search"].mean())}


# ----------------------------------------------------------------------------------------------------------- stats
def boot_w(n_p, seed=P.BOOT_SEED, n=P.N_BOOT):
    return np.random.default_rng(seed).multinomial(n_p, np.full(n_p, 1.0 / n_p), size=n).astype(np.float64)


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def pooled_diff(part, pairs, W, mask=None):
    """pairs: list of (d_a, d_b) per-token arrays over the same tokens (one per seed combo; same partition tokens,
    papers identical across bank seeds). Per-paper pooled ratio estimator + paper bootstrap."""
    m = np.ones(part.N, bool) if mask is None else mask
    sa, na, per = np.zeros(part.P), np.zeros(part.P), []
    for a, b in pairs:
        s = np.bincount(part.paper[m], weights=(a - b)[m].astype(np.float64), minlength=part.P)
        n = np.bincount(part.paper[m], minlength=part.P).astype(np.float64)
        sa += s
        na += n
        per.append(float(s.sum() / max(1, n.sum())))
    reps = (W @ sa) / np.maximum(W @ na, 1)
    return {"mean": float(sa.sum() / max(1, na.sum())), "ci": ci(reps), "per_combo": per,
            "n_tokens": int(na.sum() / max(1, len(pairs)))}


def recovery(part, dG_l, dX_l, dF_l, W, mask=None):
    m = np.ones(part.N, bool) if mask is None else mask
    num, den = np.zeros(part.P), np.zeros(part.P)
    for dG, dX, dF in zip(dG_l, dX_l, dF_l):
        num += np.bincount(part.paper[m], weights=(dG - dX)[m].astype(np.float64), minlength=part.P)
        den += np.bincount(part.paper[m], weights=(dG - dF)[m].astype(np.float64), minlength=part.P)
    reps = (W @ num) / (W @ den)
    return {"R": float(num.sum() / den.sum()), "ci": ci(reps)}
