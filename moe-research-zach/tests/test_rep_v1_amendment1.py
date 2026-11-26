"""csr-representative-v1 amendment 1: deterministic log-space bisection for tau_s / tau_r, selection tie-breaking,
both 20% gates, determinism on frozen inputs, one globally frozen temperature pair across A-F and all seeds, and the
calibration guard still unreachable from ordinary training. CPU only, CUDA hidden."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rep_calibrate_tau import (BRACKET_R, BRACKET_S, MAX_ITER, MAX_REL_GAP, S_GATE, TARGET, bisect_log,  # noqa: E402
                               calibrate_from_frozen, eff_counts, log_midpoint, select_closest)


def test_log_midpoint_deterministic():
    assert log_midpoint(0.05, 0.1) == log_midpoint(0.05, 0.1) == pytest.approx(math.sqrt(0.005), rel=1e-15)
    assert log_midpoint(2e-5, 5e-5) == pytest.approx(math.sqrt(1e-9), rel=1e-15)
    assert log_midpoint(1.0, 100.0) == pytest.approx(10.0, rel=1e-15)


def test_bracket_narrows_monotonically_and_converges():
    f = lambda t: 1000.0 * t ** 1.7  # increasing
    target = 8.0
    tr = bisect_log(f, 0.001, 1.0, target)
    assert [r["role"] for r in tr[:2]] == ["bracket_lo", "bracket_hi"] and len(tr) == 2 + MAX_ITER
    its = tr[2:]
    for a, b in zip(its, its[1:]):
        assert b["lo"] >= a["lo"] and b["hi"] <= a["hi"]
        assert math.log(b["hi"]) - math.log(b["lo"]) == pytest.approx(0.5 * (math.log(a["hi"]) - math.log(a["lo"])))
        assert a["lo"] < a["tau"] < a["hi"]
    best = select_closest(tr, target)
    exact = (target / 1000.0) ** (1 / 1.7)
    assert abs(best["tau"] / exact - 1) < 1e-4 and abs(best["value"] - target) < 1e-3


def test_no_bracket_expansion_and_early_stop_on_exact():
    tr = bisect_log(lambda t: t, 1.0, 4.0, 100.0)  # target above the bracket: converges to hi, never expands
    assert max(r["tau"] for r in tr) == 4.0 and min(r["tau"] for r in tr) == 1.0
    tr2 = bisect_log(lambda t: 2.0 if abs(t - 2.0) < 1e-12 else t, 1.0, 4.0, 2.0)
    assert tr2[-1]["role"] == "iter1" and tr2[-1]["value"] == 2.0


def test_tie_breaking_smaller_temperature():
    tr = [{"tau": 0.3, "value": 9.0}, {"tau": 0.1, "value": 7.0}, {"tau": 0.2, "value": 10.0}]
    assert select_closest(tr, 8.0)["tau"] == 0.1


def _frozen(n=60, seed=0, layers=3, scale_s=1.0, scale_g=1e-3):
    g = torch.Generator().manual_seed(seed)
    u = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    S = scale_s * (u @ u.T)
    p = torch.softmax(torch.randn(layers, n, 8, generator=g) * 0.3, dim=-1)
    G = (p @ p.transpose(1, 2)) * (scale_g / 1e-3)
    return [S], [G]


def test_gates_enforced():
    # tau_s gate: S nearly constant -> exp H(Q) ~ n-1 at every tau in [0.05, 0.1] -> outside [6.4, 9.6]
    S, G = _frozen()
    flat = [torch.zeros_like(S[0])]
    out = calibrate_from_frozen(flat, G)
    assert out["tau_s_gate_pass"] is False and out["gate_pass"] is False and "tau_s count" in out["failed_criterion"]
    assert "tau_r_trace" not in out  # stop immediately
    # tau_r gate: G constant -> exp H(R) = n-1 for all tau_r, cannot match the semantic count
    S2 = [s * 12.0 for s in S]  # sharpen semantics so the tau_s gate can pass
    out2 = calibrate_from_frozen(S2, [torch.full_like(G[0], 0.125)])
    if out2["tau_s_gate_pass"]:
        assert out2["tau_r_gate_pass"] is False and out2["relative_gap"] > MAX_REL_GAP
        assert "tau_r relative gap" in out2["failed_criterion"]
    assert S_GATE == (6.4, 9.6) and MAX_REL_GAP == 0.20 and TARGET == 8.0
    assert BRACKET_S == (0.05, 0.1) and BRACKET_R == (2e-5, 5e-5) and MAX_ITER == 20


def test_passing_case_and_repeatability_on_frozen_inputs():
    # synthetic frozen matrices whose median counts are bracketed by the prescribed brackets (checked below)
    g = torch.Generator().manual_seed(3)
    n = 100
    u = torch.nn.functional.normalize(torch.randn(n, 32, generator=g), dim=-1)
    S = [u @ u.T * 1.2]
    p = torch.softmax(torch.randn(7, n, 32, generator=g) * 0.6, dim=-1)
    G = [p @ p.transpose(1, 2) * 0.05]
    vs = [float(torch.median(eff_counts(S[0], t))) for t in BRACKET_S]
    vr = [float(torch.median(eff_counts(G[0], t).mean(0))) for t in BRACKET_R]
    assert vs[0] < TARGET < vs[1]
    a = calibrate_from_frozen(S, G)
    b = calibrate_from_frozen(S, G)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # identical repeated results
    assert a["tau_s_gate_pass"] and abs(a["selected_q_eff"] - TARGET) < 0.01
    assert vr[0] < a["selected_q_eff"] < vr[1]
    assert a["gate_pass"] and a["relative_gap"] <= 0.01 and "failed_criterion" not in a
    assert len(a["tau_s_trace"]) == len(a["tau_r_trace"]) == 2 + MAX_ITER
    for r in a["tau_s_trace"]:
        assert BRACKET_S[0] <= r["tau"] <= BRACKET_S[1] and math.isfinite(r["value"])
    for r in a["tau_r_trace"]:
        assert BRACKET_R[0] <= r["tau"] <= BRACKET_R[1] and math.isfinite(r["value"])


def test_one_global_temperature_pair_across_conditions_and_seeds():
    from rep_make_configs import base_config, build_seed_configs
    seg, rnd = ROOT / "artifacts/segments/8e7edc7e14b3c880", ROOT / "artifacts/segments_random/0c57f77e51ea3734"
    if not (seg.exists() and rnd.exists()):
        pytest.skip("rep caches absent")
    base = base_config("artifacts/segments/8e7edc7e14b3c880", "artifacts/segments_random/0c57f77e51ea3734")
    out = build_seed_configs(base, 0.0712, 3.1e-5, 0.001)
    pairs = set()
    for s, (b, cfgs) in out.items():
        assert b["init"]["seed"] == b["train"]["seed"] == s and b["data"]["train_order_seed"] == 17 + s
        for c, cfg in cfgs.items():
            pairs.add((cfg["semantic"]["tau_s"], cfg["semantic"]["tau_r"], cfg["semantic"]["lambda"]))
            assert cfg["semantic"]["enabled"] == (c in "CE")
        from csr.config import check_matched
        assert check_matched(cfgs)
    assert pairs == {(0.0712, 3.1e-5, 0.001)}
    frozen = sorted((ROOT / "configs" / "rep-v1").glob("seed*/condition_*.json"))
    if frozen:  # once generated: the on-disk configs carry exactly one pair, equal to the accepted calibration
        vals = {(json.loads(f.read_text())["semantic"]["tau_s"], json.loads(f.read_text())["semantic"]["tau_r"]) for f in frozen}
        assert len(vals) == 1 and len(frozen) == 18


def test_ordinary_training_cannot_bypass_calibration_guard():
    import inspect
    from csr.config import build_condition_config, load_json
    from csr.guard import GuardError
    from train import plan, run
    precal = ROOT / "configs" / "rep-v1" / "precal" / "base.json"
    if not precal.exists():
        pytest.skip("precal config absent")
    for c in "CE":
        cfg = build_condition_config(load_json(precal), c)
        with pytest.raises(GuardError, match="not calibrated"):
            plan(cfg)
        with pytest.raises(GuardError, match="not calibrated"):
            run(cfg, "cpu", False, "train")
    assert "require_semantic_calibration" not in inspect.signature(run).parameters
