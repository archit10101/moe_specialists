"""cfe probe metrics (numpy): topic(domain)-expert mutual information from top-4 routes, routing change between two
routings of the same tokens, paired training-loss difference against a recorded TOKEN-4 trajectory."""
import json

import numpy as np


def mi_domain(topi, dom, E=32, n_dom=5):
    out = []
    dom = dom.astype(np.int64)
    for l in range(topi.shape[1]):
        c = np.zeros((n_dom, E))
        for k in range(topi.shape[2]):
            np.add.at(c, (dom, topi[:, l, k].astype(np.int64)), 1.0)
        p = c / c.sum()
        pd, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        out.append(float((p[nz] * np.log(p[nz] / (pd @ pe)[nz])).sum()))
    return float(np.mean(out))


def routing_change(topi_a, topi_b):
    """Fraction of (token, layer) slots whose top-4 expert SET differs, and mean top-4 overlap (0..4)."""
    sa, sb = np.sort(topi_a, -1), np.sort(topi_b, -1)
    changed = float(1.0 - (sa == sb).all(-1).mean())
    sub = slice(None, None, max(1, topi_a.shape[0] // 20000))
    a_, b_ = topi_a[sub].astype(np.int64), topi_b[sub].astype(np.int64)
    ov = (a_[..., :, None] == b_[..., None, :]).any(-1).sum(-1).mean()
    return changed, float(ov)


def lm_by_step(metrics_path):
    out = {}
    with open(metrics_path) as f:
        for line in f:
            r = json.loads(line)
            if "lm" in r and "step" in r:
                out[int(r["step"])] = float(r["lm"])
    return out


def paired_dlm(own, ref, lo, hi):
    """Mean (own - ref) training LM loss over logged steps lo <= s < hi present in both (identical batches)."""
    ss = [s for s in own if lo <= s < hi and s in ref]
    if not ss:
        return None, []
    return float(np.mean([own[s] - ref[s] for s in ss])), ss
