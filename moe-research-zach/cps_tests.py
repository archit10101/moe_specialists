#!/usr/bin/env python
"""csr-context-prior-specialization-v2 CPU test suite (protocol section 12). Exit 0 iff all pass.
Writes artifacts/context-prior-specialization-v2/cpu_tests_<stamp>.json."""
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def toy_batch(B=2, T=256, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 50000, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    doc[0, 100:] = 1
    doc[1, 200:] = -1
    pos = torch.zeros(B, T, dtype=torch.long)
    for r in range(B):
        p = 0
        for t in range(T):
            if t > 0 and doc[r, t] != doc[r, t - 1]:
                p = 0
            pos[r, t] = p
            p += 1
    ids[doc < 0] = 0
    return ids, doc, pos


@test
def T01_backbone_identical_to_token4():
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.cps_model import CPSConfig, CPSModel
    a = LFRModel(LFRConfig(router="linear", teacher=False), 100).state_dict()
    for cc in (CPSConfig(teacher=True), CPSConfig(teacher=False, variant="memory"), CPSConfig(variant="token"),
               CPSConfig(chunk="adaptive", inject="late", ctx_dim=128)):
        b = CPSModel(cc, 100).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a), cc


@test
def T02_token_variant_equals_token4_forward():
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.cps_model import CPSConfig, CPSModel
    ids, doc, pos = toy_batch()
    a = LFRModel(LFRConfig(router="linear", teacher=False), 7).eval()
    b = CPSModel(CPSConfig(variant="token", teacher=False), 7).eval()
    with torch.no_grad():
        oa = a({"input_ids": ids, "doc_ids": doc, "positions": pos}, {"collect": True})
        ob = b({"input_ids": ids, "doc_ids": doc, "positions": pos}, {"collect": True})
    assert torch.equal(oa["hidden"], ob["hidden"])
    assert all(torch.equal(x["topi"], y["topi"]) for x, y in zip(oa["records"], ob["records"]))


@test
def T03_chunk_source_fixed():
    from csr.cps_model import chunk_source, fixed_boundaries
    ids, doc, pos = toy_batch()
    src = chunk_source(doc, pos, fixed_boundaries(pos, 64))
    r0 = src[0].tolist()
    assert r0[0] == -1 and r0[63] == -1 and r0[64] == 63 and r0[99] == 63
    assert r0[100] == -1 and r0[163] == -1 and r0[164] == 163 and r0[255] == 227
    assert all(v == -1 for v in src[1, 200:].tolist())
    # source index always strictly before t and in the same document
    for b in range(2):
        for t in range(256):
            s = int(src[b, t])
            if s >= 0:
                assert s < t and int(doc[b, s]) == int(doc[b, t])


@test
def T04_adaptive_boundaries_causal_and_forced():
    import torch
    from csr.cps_model import adaptive_boundaries, chunk_source, DET_DIM, DET_SEED
    ids, doc, pos = toy_batch(T=512)
    R = torch.randn(50304, DET_DIM, generator=torch.Generator().manual_seed(DET_SEED))
    b1 = adaptive_boundaries(ids, doc, pos, R, 0.8, 16)
    ids2 = ids.clone()
    ids2[:, 300:] = torch.randint(2, 50000, (2, 212))
    b2 = adaptive_boundaries(ids2, doc, pos, R, 0.8, 16)
    assert torch.equal(b1[:, :300], b2[:, :300])
    assert bool(b1[pos == 0].all()) and bool(b1[(pos % 64) == 0].all())
    src = chunk_source(doc, pos, b1)
    for b in range(2):
        for t in range(512):
            s = int(src[b, t])
            if s >= 0:
                assert s < t and int(doc[b, s]) == int(doc[b, t])


@test
def T05_model_causal_cpu_all_modes():
    import torch
    from csr.cps_model import CPSConfig, CPSModel
    ids, doc, pos = toy_batch()
    for cc in (CPSConfig(teacher=False, variant="memory"), CPSConfig(teacher=False, chunk="adaptive", det_theta=0.8),
               CPSConfig(teacher=False, shuffle_ctx=True), CPSConfig(teacher=False, variant="noresid"),
               CPSConfig(teacher=False, inject="middle", chunk_len=16)):
        m = CPSModel(cc, 3).eval()
        with torch.no_grad():
            m.cp_ab.fill_(0.0)
            m.cp_U.mul_(50.0)
            o1 = m({"input_ids": ids, "doc_ids": doc, "positions": pos}, {"collect": True})
            ids2 = ids.clone()
            ids2[:, 150:] = torch.randint(2, 50000, (2, 106))
            o2 = m({"input_ids": ids2, "doc_ids": doc, "positions": pos}, {"collect": True})
        assert all(torch.equal(x["topi"][:, :150], y["topi"][:, :150]) for x, y in zip(o1["records"], o2["records"])), cc
        assert float((o1["hidden"][:, :150] - o2["hidden"][:, :150]).abs().max()) < 1e-4, cc


@test
def T06_alpha_init_near_zero_and_routes_near_token4():
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.cps_model import CPSConfig, CPSModel
    ids, doc, pos = toy_batch()
    a = LFRModel(LFRConfig(router="linear", teacher=False), 5).eval()
    b = CPSModel(CPSConfig(teacher=False, variant="memory"), 5).eval()
    with torch.no_grad():
        oa = a({"input_ids": ids, "doc_ids": doc, "positions": pos}, {"collect": True})
        ob = b({"input_ids": ids, "doc_ids": doc, "positions": pos}, {"collect": True, "stats": True})
    al = ob["cstats"]["alpha"]
    assert all(abs(x - 0.01) < 0.002 for x in al), al
    agree = sum(float((x["topi"].sort(-1).values == y["topi"].sort(-1).values).all(-1).float().mean())
                for x, y in zip(oa["records"], ob["records"])) / len(oa["records"])
    assert agree > 0.95, agree


@test
def T07_strip_teacher():
    from csr.cps_model import CPSConfig, CPSModel
    m = CPSModel(CPSConfig(teacher=True), 0)
    assert any(n.startswith("tc_") for n, _ in m.named_parameters())
    m.strip_teacher()
    assert not any(n.startswith("tc_") for n, _ in m.named_parameters()) and not m.ccfg.teacher
    pc = m.param_counts()
    assert pc["teacher_training_only"] == 0 and pc["context_prior"] > 0


@test
def T08_docagg_matches_v1_fsi_and_paired_bootstrap():
    import numpy as np
    from csr.lfr_audit import fsi_from_deltas
    from csr.cps_audit import docagg, fsi_docagg, paired_bootstrap, specialist_test
    rng = np.random.default_rng(0)
    n, K = 5000, 5
    base = rng.gamma(2.0, 1.5, n)
    lab = rng.integers(0, K, n)
    gdoc = rng.integers(0, 120, n)
    groups = {k: ({0: [k, k + 5]} if k != 3 else {}) for k in range(K)}
    deltas = {k: rng.normal(0.01 * (lab == k), 0.05) for k in range(K) if groups[k]}
    v1 = fsi_from_deltas(base, deltas, lab, K, groups, 11)
    ag = docagg(base, deltas, lab, K, groups, 11, gdoc)
    f, I = fsi_docagg(ag)
    assert abs(f - v1["fsi"]) < 1e-9 and np.allclose(I, v1["I"])
    pb = paired_bootstrap(ag, ag, n_boot=50)
    assert abs(pb["diff"]) < 1e-12 and pb["ci95"][0] == 0.0 == pb["ci95"][1]
    st = specialist_test(ag, n_boot=100)
    assert st["significant"][3] is False and len(st["I"]) == K


@test
def T09_search_space_and_determinism():
    from csr import cps_search as S
    S.apply(S.BASE, {})
    for _, _, ch in S.WAVE_A:
        S.apply(S.BASE, ch)
    for _, _, resp in S.RULES:
        for ch in resp:
            S.apply(S.BASE, ch)
    try:
        S.apply(S.BASE, {"chunk_len": 48})
        raise AssertionError("out-of-space knob accepted")
    except ValueError:
        pass
    q = {"ce_minus_t4": 0.05, "Q3_teacher_phase": {"degraded": True}, "Q4_distillation": {"degraded": False},
         "Q2_token_flexibility": {"lost": False}, "Q1_prior_useful": {"inert": False},
         "Q6_functional": {"not_functional": False}, "Q5_specialisation": {"increased": False}}
    r1 = S.next_change(S.BASE, q, set())
    r2 = S.next_change(S.BASE, q, set())
    assert r1 == r2 == ("R1", {"lam_max": 0.5})
    tried = {S.key(S.apply(S.BASE, {"lam_max": 0.5}))}
    assert S.next_change(S.BASE, q, tried) == ("R1", {"lam_max": 0.0})
    ref = {"ce": 3.79, "fsi_domain": 0.004}
    cs = [{"id": "C01", "ce": 3.80, "fsi_domain": 0.005}, {"id": "C02", "ce": 3.78, "fsi_domain": 0.003},
          {"id": "C03", "ce": 3.85, "fsi_domain": 0.01}]
    assert S.incumbent(cs, ref)["id"] == "C01"
    assert S.advance(cs, ref) == [("C01", True)]
    assert S.advance([cs[2]], ref) == [("C03", False)]
    assert S.MAX_CANDIDATES == 12


@test
def T10_losses_gradient_paths():
    import torch
    from csr.cps_model import CPSConfig, CPSModel, spec_consistency, utility_loss, standardize
    m = CPSModel(CPSConfig(teacher=True), 0)
    n, E = 64, 32
    s = torch.randn(n, 96)
    crec = {"s": s, "valid": torch.ones(n, dtype=torch.bool)}
    cE, cS, cT = m.cf_contexts(crec, torch.tensor(0.5), False)
    u = torch.randn(n, E)
    pr = m.prior_logits(0, cE, torch.randn(n, E))
    loss = utility_loss(pr, standardize(u), "corr") + spec_consistency(pr, cE, standardize(u), u,
                                                                         torch.arange(n))[0]
    loss.backward()
    names = {nm for nm, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert names and all(nm.startswith("cp_") for nm in names), names
    for mode in ("corr", "rank", "mse"):
        assert torch.isfinite(utility_loss(torch.randn(n, E), standardize(u), mode))


@test
def T11_forward_records_and_shuffle_donor():
    import torch
    from csr.cps_model import CPSConfig, CPSModel
    ids, doc, pos = toy_batch()
    cf = torch.tensor([70, 130, 230, 256 + 70, 256 + 150])
    for sh in (False, True):
        m = CPSModel(CPSConfig(teacher=False, shuffle_ctx=sh), 1).eval()
        st = {}
        with torch.no_grad():
            m({"input_ids": ids, "doc_ids": doc, "positions": pos, "cf_idx": cf}, {"cf_store": st})
        c = st["ctx"]
        assert c["s"].shape[0] == 5 and "rec0" in st and "rec10" in st
        if not sh:
            assert c["valid"].tolist() == [True, False, True, True, True]
            assert c["gidx"][0].item() == 63
        else:  # row 0 takes row 1 (B=2 roll) contexts, row 1 takes row 0
            assert c["gidx"][0].item() == 256 + 63 and c["gidx"][3].item() == 63


@test
def T12_configs_and_cache_coverage():
    import cps_driver as drv
    from csr import cps_search as S
    from csr.cps_lib import teacher_end_ok
    for cond in ("TOKEN", "MEMORY", "CAND", "SHUF", "NORESID", "TOKEN-COMPUTE"):
        k = S.apply(S.BASE, {"chunk": "adaptive"}) if cond == "CAND" else S.BASE
        cc, c = drv.cond_cfg(cond, k)
        assert cc["teacher"] == (cond in ("CAND", "SHUF", "NORESID"))
        if cond == "CAND":
            assert cc["det_theta"] > 0
    for sch in S.SPACE["schedule"]:
        assert teacher_end_ok(sch)
    assert not teacher_end_ok("50/20/30")


@test
def T13_training_code_split_and_label_guards():
    src = (ROOT / "cps_train.py").read_text()
    assert 'ALLOWED_SPLITS = ("main", "calib_search")' in src
    assert not re.search(r'Split\("(val|test|calib_sel)"\)', src)
    assert "domain" not in src.replace("ce_dom", "").replace("per-domain", "")
    msrc = (ROOT / "src/csr/cps_model.py").read_text()
    assert '"domain"' not in msrc and "seq_id" not in msrc
    ev = (ROOT / "cps_evaluate.py").read_text()
    assert "STAGE3_OPEN.json" in ev and "frozen.json" in ev


@test
def T14_ckpt_module_is_v1_copy():
    a = (ROOT / "src/csr/lfr_ckpt.py").read_text().splitlines()
    b = (ROOT / "src/csr/cps_ckpt.py").read_text().splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and diff == [0, 13], diff


@test
def T15_evaluate_interface_cpu():
    import torch
    from csr.fsp_lib import Split
    from csr.lfr_eval import eval_rows, evaluate
    from csr.cps_model import CPSConfig, CPSModel
    sp = Split("calib_search")
    rows = eval_rows(sp, 3000)[:2]
    m = CPSModel(CPSConfig(teacher=False, variant="memory"), 0)
    torch.set_num_threads(8)
    r = evaluate(m, sp, rows, torch.device("cpu"), per_token=True, batch_rows=2)
    r2 = evaluate(m, sp, rows, torch.device("cpu"), exclude={0: [1, 2]}, per_token=True, routing=False, batch_rows=2)
    assert r["per_token"]["topi"].shape[1] == 11 and r2["ce"] >= 0
    m.force_ctx_off = True
    r3 = evaluate(m, sp, rows, torch.device("cpu"), batch_rows=2)
    assert abs(r3["ce"] - r["ce"]) < 0.5


def main():
    guard.check_root()
    import torch
    torch.set_num_threads(8)
    from csr.cps_lib import ART
    res = {"time": guard.now(), "results": {}}
    failed = []
    for f in TESTS:
        try:
            f()
            res["results"][f.__name__] = "PASS"
        except Exception:  # noqa: BLE001
            res["results"][f.__name__] = traceback.format_exc()[-1500:]
            failed.append(f.__name__)
    res["all_pass"] = not failed
    res["failed"] = failed
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, res)
    for k, v in res["results"].items():
        print(k, "PASS" if v == "PASS" else "FAIL\n" + v)
    print(json.dumps({"all_pass": res["all_pass"], "n": len(TESTS), "failed": failed,
                      "out": str(out.relative_to(ROOT))}))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
