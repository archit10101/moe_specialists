#!/usr/bin/env python
"""fspv2 CPU test suite (CUDA hidden by env.sh). Writes artifacts/forecast-specialist-pretraining-v2/cpu_tests_<stamp>.json.
Covers: causal draft rollouts / cached-feature construction, main-model causality (logits, routes, domain posteriors)
under future-token AND future-forecast perturbation, mandatory 3 general + 1 specialist execution, teacher-forced
specialist masking, exclusion re-routing, annealing schedule, sigma=0 teacher equivalence, final teacher removal,
EDL loss, data-split integrity (no overlap of paper hashes across buckets; test locked)."""
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

RESULTS = {}


def test(fn):
    def w():
        try:
            fn()
            RESULTS[fn.__name__] = "PASS"
        except Exception:
            RESULTS[fn.__name__] = "FAIL: " + traceback.format_exc()[-1500:]
    w.__name__ = fn.__name__
    return w


def tiny_cfg(**kw):
    from csr.fsp_lib import N_SCALAR
    from csr.fsp_model import FSPConfig
    base = dict(n_layer=3, d_model=64, n_head=2, dense_ffn=192, expert_ffn=32, shared_ffn=64, topic_dim=16,
                n_scalar=N_SCALAR, fc_dim=16, teacher_init=1.0)
    base.update(kw)
    return FSPConfig(**base)


def rand_batch(B=2, T=1024, seed=0, two_docs=True):
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 50000, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    pos = torch.arange(T).expand(B, T).clone()
    if two_docs:
        doc[:, 700:] = 1
        pos[:, 700:] -= 700
        doc[1, 1000:] = -1
        pos[1, 1000:] = 0
    lab = torch.randint(0, 5, (B,), generator=g).view(B, 1).expand(B, T).clone()
    lab[doc < 0] = -1
    return {"input_ids": ids, "doc_ids": doc, "positions": pos, "domain": lab,
            "seg_index": torch.zeros(B, T, dtype=torch.long)}


def fake_fc(batch, seed=0):
    import torch
    from csr.fsp_lib import build_draft, draft_rollout
    torch.manual_seed(seed)
    from csr.fsp_lib import DRAFT_CFG
    from csr.model import CSRModel, ModelConfig
    d = CSRModel(ModelConfig.from_dict({**DRAFT_CFG, "n_layer": 2, "d_model": 32, "n_head": 2, "dense_ffn": 64}))
    d.eval()
    roll, ent, top4, lp = draft_rollout(d, batch["input_ids"], batch["doc_ids"], batch["positions"], H=8)
    lp5 = torch.stack([lp * (1 + 0.1 * k) for k in range(5)], -1)
    return {"roll": roll, "ent": ent, "top": top4, "lp_a": lp5[..., 0], "lp_o": lp5[..., 1:]}


@test
def test_draft_rollout_causal():
    import torch
    b = rand_batch(B=2, T=256, two_docs=False)
    f1 = fake_fc(b)
    t = 100
    b2 = {k: v.clone() for k, v in b.items()}
    b2["input_ids"][:, t + 1:] = torch.randint(2, 50000, b2["input_ids"][:, t + 1:].shape)
    f2 = fake_fc(b2)
    for k in ("roll", "ent", "top", "lp_a"):
        assert torch.equal(f1[k][:, :t + 1], f2[k][:, :t + 1]) or torch.allclose(f1[k][:, :t + 1], f2[k][:, :t + 1],
                                                                                  atol=1e-5), k
    assert not torch.equal(f1["lp_a"][:, t + 2:], f2["lp_a"][:, t + 2:])  # sanity: later positions do change


@test
def test_rollout_is_greedy_continuation():
    """roll[t, k] equals the greedy token produced by an explicit incremental decode from the real prefix x_<=t."""
    import torch
    from csr.fsp_lib import DRAFT_CFG, draft_rollout
    from csr.model import CSRModel, ModelConfig
    torch.manual_seed(1)
    d = CSRModel(ModelConfig.from_dict({**DRAFT_CFG, "n_layer": 2, "d_model": 32, "n_head": 2, "dense_ffn": 64}))
    d.eval()
    T = 40
    ids = torch.randint(2, 50000, (1, T))
    doc = torch.zeros(1, T, dtype=torch.long)
    pos = torch.arange(T).view(1, T)
    roll, _, _, _ = draft_rollout(d, ids, doc, pos, H=4)
    from csr.model import RoutingConfig
    for t in (0, 7, 25):
        seq = ids[:, :t + 1].clone()
        for k in range(4):
            L = seq.shape[1]
            out = d({"input_ids": seq, "doc_ids": torch.zeros(1, L, dtype=torch.long),
                     "positions": torch.arange(L).view(1, L)}, RoutingConfig())
            nxt = out.logits[0, -1].argmax()
            assert int(nxt) == int(roll[0, t, k]), (t, k)
            seq = torch.cat([seq, nxt.view(1, 1)], 1)


@test
def test_causal_scalars_causal():
    import torch
    from csr.fsp_lib import causal_scalars
    b = rand_batch(B=2, T=256, two_docs=False)
    f1 = fake_fc(b)
    t = 120
    b2 = {k: v.clone() for k, v in b.items()}
    b2["input_ids"][:, t + 1:] = torch.randint(2, 50000, b2["input_ids"][:, t + 1:].shape)
    f2 = fake_fc(b2)
    s1 = causal_scalars(f1, b["input_ids"], b["doc_ids"], b["positions"], b["seg_index"])
    s2 = causal_scalars(f2, b2["input_ids"], b2["doc_ids"], b2["positions"], b2["seg_index"])
    assert torch.allclose(s1[:, :t + 1], s2[:, :t + 1], atol=1e-5)


def run_main(m, b, fc=None, sigma=0.0, forced=None, ctl_extra=None):
    import torch
    mi = {k: b[k] for k in ("input_ids", "doc_ids", "positions", "domain")}
    mi["seq_id"] = b["doc_ids"].clone()
    if "exclude_mask" in b:
        mi["exclude_mask"] = b["exclude_mask"]
    if forced is not None:
        mi["forced"] = forced
    ctl = {"use_dom": True, "collect": True}
    if fc is not None:
        from csr.fsp_lib import causal_scalars
        mi["fc"] = {"roll": fc["roll"], "scal": causal_scalars(fc, b["input_ids"], b["doc_ids"], b["positions"],
                                                                b["seg_index"])}
        mi["sigma"] = torch.tensor(sigma)
        ctl["teach"] = True
    ctl.update(ctl_extra or {})
    return m(mi, ctl)


@test
def test_main_causality_future_tokens_and_forecasts():
    import torch
    from csr.fsp_model import FSPModel
    for routing, teacher in (("spec", "spec"), ("token4", "nopart"), ("token4", "none")):
        m = FSPModel(tiny_cfg(routing=routing, teacher=teacher), 0).double()
        m.eval()
        b = rand_batch(B=2, T=256, two_docs=False)
        fc = fake_fc(b) if teacher != "none" else None
        t = 150
        b2 = {k: v.clone() for k, v in b.items()}
        b2["input_ids"][:, t + 1:] = torch.randint(2, 50000, b2["input_ids"][:, t + 1:].shape)
        fc2 = fake_fc(b2) if teacher != "none" else None
        if fc2 is not None:  # additionally scramble every cached forecast feature after t
            fc2["roll"][:, t + 1:] = torch.randint(2, 50000, fc2["roll"][:, t + 1:].shape)
            fc2["lp_o"][:, t + 1:] += 3.0
        o1 = run_main(m, b, fc, 1.0 if fc is not None else 0.0)
        o2 = run_main(m, b2, fc2, 1.0 if fc2 is not None else 0.0)
        h1, h2 = o1["hidden"][:, :t + 1], o2["hidden"][:, :t + 1]
        assert torch.allclose(h1, h2, atol=1e-9), (routing, teacher, (h1 - h2).abs().max())
        for r1, r2 in zip(o1["records"], o2["records"]):
            assert torch.equal(r1["topi"][:, :t + 1], r2["topi"][:, :t + 1])
            if "q_s" in r1:
                assert torch.allclose(r1["q_s"][:, :t + 1], r2["q_s"][:, :t + 1], atol=1e-9)


@test
def test_mandatory_three_general_one_specialist():
    import torch
    from csr.fsp_model import FSPModel, N_GENERAL
    m = FSPModel(tiny_cfg(routing="spec", teacher="spec"), 0)
    m.eval()
    b = rand_batch()
    fc = fake_fc(b)
    for sig, forced in ((1.0, b["domain"] >= 0), (0.5, torch.rand(b["domain"].shape) < 0.5), (0.0, None)):
        o = run_main(m, b, fc if sig > 0 else None, sig, forced)
        for r in o["records"]:
            ti = r["topi"].reshape(-1, 4)
            assert (ti[:, :3] < N_GENERAL).all() and (ti[:, 3] >= N_GENERAL).all() and (ti[:, 3] < 32).all()
            assert (ti.sort(-1).values.diff(dim=-1) > 0).all()  # four distinct experts
            if forced is not None:
                f = forced.reshape(-1) & (b["domain"].reshape(-1) >= 0)
                grp = (ti[:, 3] - N_GENERAL) // 2
                assert (grp[f] == b["domain"].reshape(-1)[f]).all()


@test
def test_token4_exactly_four():
    from csr.fsp_model import FSPModel
    m = FSPModel(tiny_cfg(routing="token4"), 0)
    o = run_main(m, rand_batch())
    for r in o["records"]:
        ti = r["topi"].reshape(-1, 4)
        assert (ti.sort(-1).values.diff(dim=-1) > 0).all()


@test
def test_exclusion_reroutes():
    import torch
    from csr.fsp_model import FSPModel, N_GENERAL
    for routing in ("spec", "token4"):
        m = FSPModel(tiny_cfg(routing=routing), 0)
        b = rand_batch()
        ex = {0: [N_GENERAL, N_GENERAL + 1, 3], 1: [N_GENERAL, N_GENERAL + 1, 3]}
        b["exclude_mask"] = b["doc_ids"] >= 0
        forced = (b["domain"] == 0) if routing == "spec" else None  # forced group 0 is excluded -> free choice
        o = run_main(m, b, forced=forced, ctl_extra={"exclude": ex})
        for li, r in enumerate(o["records"]):
            ti = r["topi"].reshape(-1, 4)[b["exclude_mask"].reshape(-1)]
            if li in ex:
                for e in ex[li]:
                    assert not (ti == e).any()
            if routing == "spec":
                assert (ti[:, 3] >= N_GENERAL).all()
            assert torch.isfinite(r["gates"]).all() and (r["gates"] >= 0).all()


@test
def test_sigma_zero_equals_student():
    import torch
    from csr.fsp_model import FSPModel
    m = FSPModel(tiny_cfg(routing="spec", teacher="spec"), 0).double()
    b = rand_batch()
    fc = fake_fc(b)
    o1 = run_main(m, b, fc, 0.0)
    o2 = run_main(m, b, None, 0.0)
    assert torch.allclose(o1["hidden"], o2["hidden"], atol=1e-10)


@test
def test_anneal_schedule():
    from csr.fsp_lib import curriculum
    assert curriculum(0.0) == (1.0, 1.0, 1) and curriculum(0.399) == (1.0, 1.0, 1)
    s, t, p = curriculum(0.45)
    assert abs(s - 0.5) < 1e-9 and abs(t - 0.5) < 1e-9 and p == 2
    s, t, p = curriculum(0.425, "cosine")
    assert abs(s - 0.5 * (1 + __import__("math").cos(__import__("math").pi * 0.25))) < 1e-9
    assert curriculum(0.5) == (0.0, 0.0, 3) and curriculum(1.0) == (0.0, 0.0, 3)
    xs = [curriculum(0.4 + i / 1000)[0] for i in range(101)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


@test
def test_strip_teacher_inference():
    import torch
    from csr.fsp_model import FSPModel, TEACHER_PREFIX
    m = FSPModel(tiny_cfg(routing="spec", teacher="spec"), 0)
    n_before = m.param_counts()
    m.strip_teacher()
    assert not any(n.startswith(TEACHER_PREFIX) for n, _ in m.named_parameters())
    assert not any(k.startswith(TEACHER_PREFIX) for k in m.state_dict())
    b = rand_batch()
    mi = {k: b[k] for k in ("input_ids", "doc_ids", "positions")}  # no labels, no forecasts
    o = m(mi, {"use_dom": True})
    assert torch.isfinite(o["hidden"]).all()
    assert m.param_counts()["total_inference"] == n_before["total_inference"]


@test
def test_param_parity_across_conditions():
    from csr.fsp_model import FSPConfig, FSPModel
    counts = {}
    for routing, teacher in (("token4", "none"), ("spec", "spec"), ("spec", "none"), ("token4", "nopart")):
        m = FSPModel(FSPConfig(routing=routing, teacher=teacher, n_scalar=21), 0)
        counts[(routing, teacher)] = (m.param_counts()["total_inference"], m.param_counts()["active_per_token"])
    assert len(set(counts.values())) == 1, counts
    tot, act = list(counts.values())[0]
    assert 100e6 <= tot <= 115e6 and 38e6 <= act <= 45e6, (tot, act)


@test
def test_identical_init_across_conditions():
    import torch
    from csr.fsp_model import FSPModel
    a = FSPModel(tiny_cfg(routing="token4", teacher="none"), 7).inference_params()
    b = FSPModel(tiny_cfg(routing="spec", teacher="spec"), 7).inference_params()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


@test
def test_edl_loss():
    import torch
    from csr.fsp_model import edl_loss
    N, E = 400, 8
    lab = torch.arange(N) // 100 % 2
    sid = torch.arange(N) // 50
    p_same = torch.softmax(torch.zeros(N, E), -1)
    l_same = edl_loss(p_same, lab, sid, torch.ones(N, dtype=torch.bool))
    lg = torch.zeros(N, E)
    lg[lab == 0, 0] = 3
    lg[lab == 1, 1] = 3
    l_div = edl_loss(torch.softmax(lg, -1), lab, sid, torch.ones(N, dtype=torch.bool))
    assert l_same > l_div and abs(float(l_same) - (-torch.log(torch.tensor(1e-8)).item())) < 1e-3
    assert float(edl_loss(p_same, torch.zeros(N, dtype=torch.long), sid, torch.ones(N, dtype=torch.bool))) == 0.0


@test
def test_data_integrity():
    import numpy as np
    from csr.fsp_lib import DATA, OPEN_SPLITS, Split
    man = json.loads((DATA / "manifest.json").read_text())
    seen = {}
    for b in list(OPEN_SPLITS):
        d = json.loads((DATA / f"{b}_docs.json").read_text())
        for p in set(d["papers"]):
            assert p not in seen, (p, seen.get(p), b)
            seen[p] = b
    # the locked test bucket is checked by its manifest hash only (never decoded here): papers listed in its docs
    # json are compared by hash against the open buckets
    td = json.loads((DATA / "locked" / "test_docs.json").read_text())
    assert not (set(td["papers"]) & set(seen)), "test paper in an open bucket"
    try:
        Split("test")
        raise AssertionError("test split loadable")
    except guard.GuardError:
        pass
    for b in OPEN_SPLITS:
        assert guard.sha256_file(DATA / f"{b}_tokens.npy") == man["splits"][b]["sha256_tokens"]
    q = man["quotas_valid_tokens_per_domain"]
    sp = Split("main")
    per = np.bincount(sp.domains, weights=sp.lengths - 1, minlength=5)
    assert (per == q["main"]).all(), per
    # wrong / shuffled maps
    assert (sp.wrong != sp.domains).all()
    frac_same = float((sp.shuffled == sp.domains).mean())
    assert 0.1 < frac_same < 0.3, frac_same


def main():
    guard.check_root()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise SystemExit("CPU suite must run with CUDA hidden (source env.sh)")
    import torch
    torch.set_num_threads(4)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, RESULTS[name][:300], flush=True)
    ok = all(v == "PASS" for v in RESULTS.values())
    p = guard.ROOT / f"artifacts/forecast-specialist-pretraining-v2/cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, {"time": guard.now(), "all_pass": ok, "results": RESULTS})
    print("ALL PASS" if ok else "FAILURES", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
