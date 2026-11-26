"""csr-centered-fedavg-residual-v1 (cfar1): federated residual-specialist banks (follow-up to csur1).

FedBank = csur1 UtilBank machinery (same experts module 384->256->384, same causal GRU student router, same detached
residual objective) with HARD group assignment: the group of view v is argmax of its region prior (semantic p_sem or
the persistent transplant p_SH). Live specialist k trains locally on the views of group k ("clients"). The student
router is trained by KL(onehot(group) || q) and deploys argmax q.

Hidden units [0, w_p) of every specialist are PERSONALISED, units [w_p, 256) are SHARED-ONLY.

mode 'local'    : no federation (independent specialists; the no-transfer reference).
mode 'fedavg'   : F1 standard FedAvg. All specialists start identical (expert 0's init). Every T steps the
                  token-mass-weighted mean update since the last synchronisation is applied to the common anchor and
                  broadcast to every specialist (all specialists identical after each sync).
mode 'centered' : F2 centered personalised FedAvg, residual_k = S + P_k (parameter space; P_k = 0 on shared-only units).
                  Every T steps: Dbar = token-weighted mean of the local updates; S <- S + Dbar (transfer of the common
                  component); personalised units: P_k <- P_k + D_k - Dbar, so S + P_k is unchanged (function continuous
                  at synchronisation); shared-only units: every specialist <- S (FedAvg on shared-only units).
Optimizer moments: personalised units of a centered bank keep their moments (parameters unchanged); every reset
(F1 all units, F2 shared-only units) sets the AdamW moments of the reset slices to their token-weighted mean over
specialists (consistent with averaging the parameters). Deployable parameters: fold S + P_k (exactly the live tensors)."""
import torch

from csr.csur1_bank import UtilBank
from csr.csur1_paths import K


class FedBank(UtilBank):
    """spec: name, kind 'fed', prior ('sem'|'shuf'), mode ('local'|'fedavg'|'centered'), T (interval), w_p."""

    def __init__(self, spec, seed):
        s = dict(spec)
        s.setdefault("util", False)
        s.setdefault("eta", 1.0)
        s.setdefault("lam", 0.0)
        super().__init__(s, seed)
        self.spec = dict(spec)
        self.mode = spec["mode"]
        self.T = int(spec.get("T", 0))
        self.w_p = int(spec.get("w_p", 256))
        f = self.experts.w1.shape[2]
        with torch.no_grad():
            if self.mode == "fedavg":
                self.experts.w1.copy_(self.experts.w1[0:1].expand_as(self.experts.w1))
                self.experts.w2.copy_(self.experts.w2[0:1].expand_as(self.experts.w2))
            elif self.mode == "centered":
                sh = slice(self.w_p, f)
                self.experts.w1[:, :, sh] = self.experts.w1[:, :, sh].mean(0, keepdim=True)
                self.experts.w2[:, sh, :] = self.experts.w2[:, sh, :].mean(0, keepdim=True)
        self.register_buffer("S1", self.experts.w1.detach().mean(0).clone())
        self.register_buffer("S2", self.experts.w2.detach().mean(0).clone())
        self.register_buffer("A1", self.experts.w1.detach().clone())  # local anchors at the last synchronisation
        self.register_buffer("A2", self.experts.w2.detach().clone())
        self.register_buffer("mass", torch.zeros(K))
        self.register_buffer("n_sync", torch.zeros((), dtype=torch.long))

    def n_params(self):
        f = self.experts.w1.shape[2]
        d = self.experts.w1.shape[1]
        r = sum(p.numel() for m in (self.inp, self.gru, self.head) for p in m.parameters())
        if self.mode == "centered":
            unique = 2 * d * f + K * 2 * d * self.w_p  # S (all units) + P_k (personalised units)
            deploy = 2 * d * (f - self.w_p) + K * 2 * d * self.w_p
        elif self.mode == "fedavg":
            unique = deploy = K * 2 * d * f
        else:
            unique = deploy = K * 2 * d * f
        return {"experts": deploy, "experts_training_unique": unique, "router": r}

    def responsibilities(self, u, partner, logprior, update=False, weights=None, u_w=None):
        """Hard groups: r = onehot(argmax prior)."""
        g = logprior.argmax(1)
        r = torch.nn.functional.one_hot(g, K).float()
        if update:
            w = weights.float() if weights is not None else torch.ones(r.shape[0], device=r.device)
            self.mass += (r * w.unsqueeze(1)).sum(0)
            self.n_upd += 1
        return r, {"r_ent": 0.0, "agree_prior": 1.0}

    @torch.no_grad()
    def sync(self, opt):
        """Federation at the end of an interval (after the optimizer step). Returns diagnostics."""
        if self.mode == "local" or self.T <= 0:
            return None
        w = self.mass / self.mass.sum().clamp(min=1)
        W1, W2 = self.experts.w1, self.experts.w2
        D1, D2 = W1 - self.A1, W2 - self.A2
        Db1 = torch.einsum("k,kdf->df", w, D1)
        Db2 = torch.einsum("k,kfd->fd", w, D2)
        f = W1.shape[2]
        before = (W1.clone(), W2.clone())
        st = opt.state
        if self.mode == "fedavg":
            G1, G2 = self.A1[0] + Db1, self.A2[0] + Db2
            W1.copy_(G1.unsqueeze(0).expand_as(W1))
            W2.copy_(G2.unsqueeze(0).expand_as(W2))
            reset1, reset2 = (slice(None), slice(None), slice(None)), (slice(None), slice(None), slice(None))
        else:
            self.S1 += Db1
            self.S2 += Db2
            sh = slice(self.w_p, f)
            W1[:, :, sh] = self.S1[:, sh].unsqueeze(0)
            W2[:, sh, :] = self.S2[sh, :].unsqueeze(0)
            reset1, reset2 = (slice(None), slice(None), sh), (slice(None), sh, slice(None))
        for p, rs in ((W1, reset1), (W2, reset2)):
            s_ = st.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in s_:
                    m = s_[key]
                    avg = torch.einsum("k,k...->...", w, m[rs])
                    m[rs] = avg.unsqueeze(0).expand_as(m[rs])
        cont = {}
        if self.mode == "centered":
            P = slice(0, self.w_p)
            cont = {"cont_max_abs_personalised": float(max((W1[:, :, P] - before[0][:, :, P]).abs().max(),
                                                           (W2[:, P, :] - before[1][:, P, :]).abs().max()))}
        self.A1.copy_(W1)
        self.A2.copy_(W2)
        self.mass.zero_()
        self.n_sync += 1
        spread = float((W1 - W1.mean(0, keepdim=True)).norm() / W1.norm().clamp(min=1e-12))
        return {"sync": int(self.n_sync), "dbar_norm": float(Db1.norm() + Db2.norm()),
                "local_upd_norm": float(D1.norm() + D2.norm()), "specialist_spread": spread, **cont}
