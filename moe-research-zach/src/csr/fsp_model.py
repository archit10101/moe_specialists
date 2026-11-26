"""csr-forecast-specialist-pretraining-v2 (fspv2) models. New module; nothing existing is modified.

Main model (all conditions share the identical inference-time parameter set):
  12 layers, d=384, 6 heads (RoPE), layer 0 dense GELU FFN 1792 (= shared 768 + 4 x 256: same active FFN width as the
  MoE layers), layers 1..11 MoE: 32 routed GELU experts (FFN 256) + one always-active shared GELU expert (FFN 768),
  tied embeddings (V=50304). Per MoE layer the inference model also owns the causal topic/domain module
  (topic projection 384->64 with a fixed-rate causal EMA, domain head 5x(64+384)); it is used by every condition
  except TOKEN-4/EDL, where it exists but receives no gradient (identical total parameter count).

Routing modes
  token4 : FLAME routing. p = softmax(l) over 32; top-4 by p; gates = p (no renormalisation).
  spec   : experts 0..21 general, 22..31 specialists; domain d owns specialists {22+2d, 23+2d}.
           Every token executes top-3 general (by p over the 22) + exactly one specialist. Gates = joint softmax
           of the (teacher-modified) logits over all 32 experts, as in token4: only the eligible sets differ.
           Specialist choice: score_e = l'_e + beta * log q(dom(e)), argmax over the 10 specialists, where
           q = teacher/student domain posterior; teacher-forced tokens are masked to the label's group.
Teacher (training only, removed at inference; prefix "tc_"): forecast encoder c = Enc(cached draft features) and
  per-layer residuals A_l c (10 specialist logits, or 32 for FORECAST-NOPART) and G_l c (5 domain logits),
  scaled by sigma = 1 - lambda (anneal). At sigma = 0 and tf = 0 the forward is exactly the standalone student.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm, apply_rope, rope_cos_sin

N_DOMAINS = 5
N_GENERAL = 22
N_SPEC = 10
TEACHER_PREFIX = "tc_"


def spec_group(d):
    return [N_GENERAL + 2 * d, N_GENERAL + 2 * d + 1]


@dataclass
class FSPConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    d_model: int = 384
    n_head: int = 6
    dense_ffn: int = 1792
    n_experts: int = 32
    top_k: int = 4
    expert_ffn: int = 256
    shared_ffn: int = 768
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    topic_dim: int = 64
    n_domains: int = N_DOMAINS
    # routing / teacher (condition + engineering knobs; none change the inference parameter set)
    routing: str = "token4"  # token4 | spec
    topic_halflife: float = 128.0  # tokens; EMA rate of the causal topic state
    spec_beta: float = 1.0  # weight of log q(domain) in the specialist score
    teacher: str = "none"  # none | spec | nopart  (forecast teacher residual target)
    fc_dim: int = 64  # forecast code width
    fc_horizon: int = 8  # rollout tokens used by the forecast encoder (1, 4 or 8)
    n_scalar: int = 0  # filled by the feature builder
    teacher_init: float = 0.0  # init std multiplier of the residual output weights (0 = start at the student)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class GELUMLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], 2)
                v = torch.cat([kv_cache["v"], v], 2)
            kv_cache["k"], kv_cache["v"] = k, v
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Experts(nn.Module):
    def __init__(self, E, d, f):
        super().__init__()
        self.E = E
        self.w1 = nn.Parameter(torch.empty(E, d, f))
        self.w2 = nn.Parameter(torch.empty(E, f, d))

    @torch.compiler.disable
    def forward(self, x, topi, gates):
        """Dropless sorted dispatch, one matmul pair per non-empty expert. x (N, d); topi/gates (N, k)."""
        N, d = x.shape
        k = topi.shape[1]
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // k
        xs = x.index_select(0, tok)
        w1s, w2s = self.w1.to(x.dtype).unbind(0), self.w2.to(x.dtype).unbind(0)
        parts = torch.split(xs, counts)
        outs = [F.gelu(parts[e] @ w1s[e], approximate="tanh") @ w2s[e] for e, c in enumerate(counts) if c]
        ys = torch.cat(outs, 0) * gates.reshape(-1)[order].to(x.dtype).unsqueeze(-1)
        out = torch.empty_like(ys)
        out[order] = ys  # un-sort (deterministic; no atomics)
        return out.view(N, k, d).sum(1)


class MoELayer(nn.Module):
    def __init__(self, cfg: FSPConfig):
        super().__init__()
        d, E = cfg.d_model, cfg.n_experts
        self.router = nn.Linear(d, E, bias=False)
        self.experts = Experts(E, d, cfg.expert_ffn)
        self.shared = GELUMLP(d, cfg.shared_ffn)
        # causal topic / domain module (inference-time; identical in every condition)
        self.topic_proj = nn.Linear(d, cfg.topic_dim, bias=False)
        self.topic_norm = RMSNorm(cfg.topic_dim, cfg.norm_eps)
        self.dom_topic = nn.Linear(cfg.topic_dim, cfg.n_domains, bias=False)
        self.dom_tok = nn.Linear(d, cfg.n_domains, bias=False)


class Block(nn.Module):
    def __init__(self, cfg, is_moe):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = is_moe
        self.ffn = MoELayer(cfg) if is_moe else GELUMLP(cfg.d_model, cfg.dense_ffn)


def ema_matrix(doc_ids, positions, halflife, dtype):
    """(B, T, T) row-normalised causal EMA weights inside each document: w[t, s] ∝ r^(t-s), s <= t, same doc."""
    B, T = doc_ids.shape
    r = 0.5 ** (1.0 / halflife)
    t = torch.arange(T, device=doc_ids.device)
    lag = (t.view(T, 1) - t.view(1, T)).float()
    base = torch.where(lag >= 0, torch.pow(torch.tensor(r, device=doc_ids.device), lag.clamp(min=0)),
                       torch.zeros((), device=doc_ids.device))
    same = (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)) & (doc_ids >= 0).unsqueeze(2)
    w = base.unsqueeze(0) * same
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
    return w.to(dtype)


@dataclass
class Out:
    logits: torch.Tensor
    aux: torch.Tensor  # load balance (sum over MoE layers)
    z: torch.Tensor  # router z-loss (sum over layers)
    dom_ce: torch.Tensor  # domain CE of the student heads (mean over layers), 0 if unused
    dom_ce_teacher: torch.Tensor
    distill: torch.Tensor  # routing distillation KL (mean over layers)
    dom_kl: torch.Tensor  # teacher->student domain KL (mean over layers)
    edl: torch.Tensor  # EDL loss (mean over layers)
    records: list = field(default_factory=list)


class FSPModel(nn.Module):
    def __init__(self, cfg: FSPConfig, seed: int):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(seed)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i > 0) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.reset_parameters()
        self.n_moe = cfg.n_layer - 1
        # teacher (training only)
        if cfg.teacher != "none":
            g = torch.Generator().manual_seed(1_000_003 + seed)
            d, fc = cfg.d_model, cfg.fc_dim
            self.tc_hpos = nn.Parameter(torch.randn(8, d, generator=g) * cfg.init_std)
            self.tc_in = nn.Linear(d + cfg.n_scalar, 2 * fc, bias=True)
            self.tc_out = nn.Linear(2 * fc, fc, bias=True)
            self.tc_norm = RMSNorm(fc, cfg.norm_eps)
            n_out = N_SPEC if cfg.teacher == "spec" else cfg.n_experts
            self.tc_layer = nn.Parameter(torch.randn(self.n_moe, fc, generator=g) * cfg.init_std)  # layer identity
            self.tc_route = nn.Parameter(torch.randn(self.n_moe, n_out, fc, generator=g) * cfg.init_std
                                         * max(cfg.teacher_init, 1e-3))
            self.tc_dom = nn.Parameter(torch.randn(self.n_moe, cfg.n_domains, fc, generator=g) * cfg.init_std)
            with torch.no_grad():
                for lin in (self.tc_in, self.tc_out):
                    lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * cfg.init_std)
                    lin.bias.zero_()

    @torch.no_grad()
    def reset_parameters(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2.0 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.dim() == 1:
                p.fill_(1.0)
            elif name.endswith(("attn.proj.weight", "w2.weight", "experts.w2")):
                p.normal_(0.0, out_std)
            else:
                p.normal_(0.0, std)

    # ------------------------------------------------ bookkeeping
    def inference_params(self):
        return {n: p for n, p in self.named_parameters() if not n.startswith(TEACHER_PREFIX)}

    def param_counts(self):
        cfg = self.cfg
        inf = sum(p.numel() for p in self.inference_params().values())
        teacher = sum(p.numel() for n, p in self.named_parameters() if n.startswith(TEACHER_PREFIX))
        per_expert = 2 * cfg.d_model * cfg.expert_ffn
        inactive = self.n_moe * (cfg.n_experts - cfg.top_k) * per_expert
        topic = sum(p.numel() for n, p in self.named_parameters() if any(
            k in n for k in ("topic_proj", "topic_norm", "dom_topic", "dom_tok")))
        emb = self.tok_emb.weight.numel()
        return {"total_inference": inf, "active_per_token": inf - inactive, "teacher_training_only": teacher,
                "topic_domain_module": topic, "embedding_tied": emb,
                "active_non_embedding": inf - inactive - emb}

    def strip_teacher(self):
        """Inference export: drop every teacher parameter (drafts never enter the model)."""
        for n in sorted({n.split(".")[0] for n, _ in self.named_parameters() if n.startswith(TEACHER_PREFIX)}):
            delattr(self, n)
        self.cfg = FSPConfig.from_dict({**self.cfg.to_dict(), "teacher": "none"})
        return self

    @staticmethod
    def attention_mask(doc_ids):
        T = doc_ids.shape[1]
        same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
        m = same & causal & (doc_ids >= 0).unsqueeze(2)
        pad_self = (doc_ids < 0).unsqueeze(2) & torch.eye(T, dtype=torch.bool, device=doc_ids.device)
        return (m | pad_self).unsqueeze(1)

    @torch.compiler.disable
    def forecast_code(self, fc):
        """fc: dict with 'roll' (B,T,8) long rollout ids of the assigned draft, 'scal' (B,T,S) float scalars.
        Returns c (B,T,fc_dim). Rollout embeddings use the (detached) tied embedding."""
        H = self.cfg.fc_horizon
        rd = torch.float64 if self.tok_emb.weight.dtype == torch.float64 else torch.float32
        e = self.tok_emb(fc["roll"][..., :H]).detach() + self.tc_hpos[:H].to(self.tok_emb.weight.dtype)
        e = e.mean(-2)
        x = torch.cat([e.to(rd), fc["scal"].to(rd)], -1)
        # eager (compiler-disabled) fp32: inductor mis-typed this autocast-disabled block in training graphs
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = F.gelu(F.linear(x, self.tc_in.weight.to(rd), self.tc_in.bias.to(rd)), approximate="tanh").to(rd)
            c = self.tc_norm(F.linear(h, self.tc_out.weight.to(rd), self.tc_out.bias.to(rd)))
        return c

    # ------------------------------------------------ forward
    def forward(self, batch, ctl=None):
        """batch tensors (B,T): input_ids, doc_ids, positions (long); optional 'domain' (long, -1 = none) = routing /
        auxiliary label; 'forced' (bool) = tokens whose specialist group is teacher-forced to 'domain'; 'fc' dict of
        cached draft features; 'sigma' 0-d float tensor (teacher strength); 'seq_id' (long) for EDL.
        ctl (static python flags): teach, use_dom, edl, collect, single_doc, exclude {moe_idx: [experts]} with
        batch['exclude_mask'] (B,T) bool, swap {moe_idx: perm}. Returns dict (hidden = final normed state)."""
        ctl = ctl or {}
        cfg = self.cfg
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        dev = ids.device
        N = B * T
        rd = torch.float64 if self.tok_emb.weight.dtype == torch.float64 else torch.float32
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = None if ctl.get("single_doc") else self.attention_mask(doc_ids)
        nonpad = (doc_ids >= 0).reshape(-1)
        npf = nonpad.to(rd)
        n_valid = npf.sum().clamp(min=1)
        spec = cfg.routing == "spec"
        teach = bool(ctl.get("teach")) and cfg.teacher != "none"
        use_dom = bool(ctl.get("use_dom"))
        has_lab = "domain" in batch
        lab = batch["domain"].reshape(-1) if has_lab else None
        labc = lab.clamp(min=0) if has_lab else None
        labm = ((lab >= 0) & nonpad).to(rd) if has_lab else None
        n_lab = labm.sum().clamp(min=1) if has_lab else None
        forced = batch["forced"].reshape(-1) if (spec and "forced" in batch) else None
        c = None
        if teach:
            sigma = batch["sigma"].to(rd)
            c = self.forecast_code(batch["fc"]).reshape(N, -1)
        W = ema_matrix(doc_ids, positions, cfg.topic_halflife, rd) if use_dom else None
        excl = ctl.get("exclude") or {}
        swap = ctl.get("swap") or {}
        z0 = torch.zeros((), device=dev, dtype=rd)
        aux, zl, dce, dce_t, dist, dkl, edl = z0, z0, z0, z0, z0, z0, z0
        records = []
        grp_of = torch.arange(N_SPEC, device=dev) // 2
        mi = 0
        for blk in self.blocks:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(N, -1)
            with torch.autocast(device_type=dev.type, enabled=False):
                hf32 = hf.to(rd)
                logits = F.linear(hf32, moe.router.weight.to(rd))  # student router logits (N, 32)
                lt = logits
                q_s = q_t = None
                if use_dom:
                    tp = F.linear(hf32, moe.topic_proj.weight.to(rd)).view(B, T, -1)
                    u = torch.bmm(W, tp).reshape(N, -1)
                    dl_s = F.linear(moe.topic_norm(u), moe.dom_topic.weight.to(rd)) + \
                        F.linear(hf32, moe.dom_tok.weight.to(rd))
                    q_s = torch.log_softmax(dl_s, -1)
                    q_t = q_s
                    if teach:
                        q_t = torch.log_softmax(dl_s + sigma * F.linear(c + self.tc_layer[mi], self.tc_dom[mi]), -1)
                    if has_lab:
                        dce = dce + (-(q_s.gather(1, labc.view(-1, 1)).squeeze(1)) * labm).sum() / n_lab
                        if teach:
                            dce_t = dce_t + (-(q_t.gather(1, labc.view(-1, 1)).squeeze(1)) * labm).sum() / n_lab
                            qtd = q_t.detach()
                            dkl = dkl + ((qtd.exp() * (qtd - q_s)).sum(-1) * labm).sum() / n_lab
                if teach:
                    res = sigma * F.linear(c + self.tc_layer[mi], self.tc_route[mi])
                    if cfg.teacher == "spec":
                        lt = torch.cat([logits[:, :N_GENERAL], logits[:, N_GENERAL:] + res], -1)
                    else:
                        lt = logits + res
                kill = None
                if mi in excl:
                    em = torch.zeros(cfg.n_experts, dtype=torch.bool, device=dev)
                    em[list(excl[mi])] = True
                    kill = batch["exclude_mask"].reshape(-1, 1) & em.view(1, -1)
                    lt = lt.masked_fill(kill, float("-inf"))
                p = torch.softmax(lt, -1)
                if not spec:
                    gates, topi = p.topk(cfg.top_k, -1)
                    if teach:
                        lpt = torch.log_softmax(lt.detach(), -1)
                        lps = torch.log_softmax(logits, -1)
                        dist = dist + ((lpt.exp() * (lpt - lps)).sum(-1) * npf).sum() / n_valid
                    cnt = F.one_hot(topi, cfg.n_experts).to(rd).sum(1)  # (N, E)
                    f = (cnt * npf.view(-1, 1)).sum(0) / (n_valid * cfg.top_k)
                    P = (p * npf.view(-1, 1)).sum(0) / n_valid
                    aux = aux + cfg.n_experts * (f * P).sum()
                else:
                    g3, i3 = p[:, :N_GENERAL].topk(3, -1)
                    sc = lt[:, N_GENERAL:]
                    if q_t is not None:
                        sc = sc + cfg.spec_beta * q_t.detach().repeat_interleave(2, dim=1)
                    if forced is not None:
                        fm = forced.view(-1, 1) & (grp_of.view(1, -1) != labc.view(-1, 1))
                        sc = sc.masked_fill(fm, float("-inf"))
                    if kill is not None:
                        sc_free = lt[:, N_GENERAL:]
                        sc = sc.masked_fill(kill[:, N_GENERAL:], float("-inf"))
                        dead = torch.isinf(sc).all(-1, keepdim=True)  # forced group excluded -> free choice
                        sc = torch.where(dead, sc_free, sc)
                    si = sc.argmax(-1, keepdim=True)
                    topi = torch.cat([i3, si + N_GENERAL], -1)
                    gates = torch.cat([g3, p[:, N_GENERAL:].gather(1, si)], -1)
                    gsel = si.squeeze(1) // 2
                    cols = torch.stack([N_GENERAL + 2 * gsel, N_GENERAL + 2 * gsel + 1], 1)
                    if teach:  # within-group teacher -> student distillation (teacher detached)
                        lt_w = torch.log_softmax(lt.gather(1, cols).detach(), -1)
                        ls_w = torch.log_softmax(logits.gather(1, cols), -1)
                        dist = dist + ((lt_w.exp() * (lt_w - ls_w)).sum(-1) * npf).sum() / n_valid
                    # general-bank Switch balance over the 22 general experts
                    pg = torch.softmax(lt[:, :N_GENERAL], -1)
                    cnt = F.one_hot(i3, N_GENERAL).to(rd).sum(1)
                    f = (cnt * npf.view(-1, 1)).sum(0) / (n_valid * 3)
                    P = (pg * npf.view(-1, 1)).sum(0) / n_valid
                    aux = aux + N_GENERAL * (f * P).sum()
                    # within-group balance of the two specialists of each group (token-share weighted)
                    pw = torch.softmax(lt.gather(1, cols), -1) * npf.view(-1, 1)  # (N, 2)
                    oh = F.one_hot(si.squeeze(1) % 2, 2).to(rd) * npf.view(-1, 1)
                    gm = F.one_hot(gsel, N_DOMAINS).to(rd)  # (N, 5)
                    ng = gm.t() @ npf  # (5,)
                    fw = gm.t() @ oh / ng.clamp(min=1).view(-1, 1)
                    Pw = gm.t() @ pw / ng.clamp(min=1).view(-1, 1)
                    aux = aux + ((ng / n_valid) * 2 * (fw * Pw).sum(-1)).sum()
                zl = zl + (torch.logsumexp(lt, -1).pow(2) * npf).sum() / n_valid
                if ctl.get("edl") and has_lab:
                    edl = edl + edl_loss(torch.softmax(logits, -1), lab, batch["seq_id"].reshape(-1), nonpad)
            topi_exec = topi
            if mi in swap:
                topi_exec = torch.as_tensor(swap[mi], device=dev)[topi]
            y = moe.experts(hf, topi_exec, gates) + moe.shared(hf)
            x = x + y.view(B, T, -1)
            rec = {"topi": topi.view(B, T, -1)}
            if ctl.get("collect"):
                rec["gates"] = gates.view(B, T, -1)
                rec["p"] = p.view(B, T, -1)
                if q_s is not None:
                    rec["q_s"] = q_s.view(B, T, -1)
            records.append(rec)
            mi += 1
        L = float(self.n_moe)
        return {"hidden": self.norm_f(x), "aux": aux, "z": zl, "dom_ce": dce / L, "dom_ce_teacher": dce_t / L,
                "distill": dist / L, "dom_kl": dkl / L, "edl": edl / L, "records": records}

    def lm_logits(self, hidden):
        return F.linear(hidden, self.tok_emb.weight)


@torch.compiler.disable
def edl_loss(p, lab, seq_id, nonpad, eps=1e-8):
    """Expert Divergence Learning (Li et al., ICLR 2026, Eq. 5-8): token router probs -> per-sequence mean ->
    per-domain mean of sequence means -> mean over present domain pairs of -log(JSD + eps). p: (N, E) full softmax."""
    m = nonpad & (lab >= 0) & (seq_id >= 0)
    p, lab, sid = p[m], lab[m], seq_id[m]
    if p.shape[0] == 0:
        return p.new_zeros(())
    S = int(sid.max()) + 1
    E = p.shape[1]
    ssum = p.new_zeros(S, E).index_add_(0, sid, p)
    scnt = p.new_zeros(S).index_add_(0, sid, torch.ones_like(sid, dtype=p.dtype))
    sdom = torch.full((S,), -1, dtype=torch.long, device=p.device).scatter_(0, sid, lab)
    ok = scnt > 0
    smean = ssum[ok] / scnt[ok].unsqueeze(1)
    sdom = sdom[ok]
    doms = torch.unique(sdom)
    if doms.numel() < 2:
        return p.new_zeros(())
    P = torch.stack([smean[sdom == d].mean(0) for d in doms])  # (M, E)
    M = P.shape[0]
    a, b = torch.triu_indices(M, M, 1, device=p.device)
    pa, pb = P[a], P[b]
    mm = 0.5 * (pa + pb)
    kl = lambda x, y: (x * (torch.log(x.clamp(min=1e-12)) - torch.log(y.clamp(min=1e-12)))).sum(-1)
    jsd = 0.5 * kl(pa, mm) + 0.5 * kl(pb, mm)
    return (-torch.log(jsd + eps)).mean()


MAIN_CFG = FSPConfig()
