"""Calibration/plan guard interface and top-level result_label provenance (CPU, CUDA hidden, TINY model).

- ordinary train/benchmark planning of a semantic-loss condition fails closed while lambda/tau_s/tau_r are unset;
- only calibrate.py may plan with require_semantic_calibration=False, and the bypass is not reachable from
  train/benchmark entry points;
- calibrate.py completes end to end on a tiny CPU fixture with the semantic values unset;
- calibration, evaluation and probe outputs carry the resolved run config's result_label at top level.
Writes under .tmp/test_runs only. Pipeline checks, no scientific claim."""
import inspect
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SMOKE_BASE = ROOT / "configs" / "smoke-small-v1" / "base.json"
pytestmark = pytest.mark.skipif(not SMOKE_BASE.exists(), reason="smoke config not generated")

TINY = {"model.d_model": 32, "model.n_head": 4, "model.dense_ffn": 48, "model.n_layer": 3, "model.moe_layers": [1, 2],
        "model.n_experts": 8, "model.top_k": 2, "model.expert_ffn": 16, "model.shared_ffn": 24,
        "routing.pool_size": 4, "train.micro_batch_rows": 2, "train.token_budget": 3000, "train.warmup_steps": 1,
        "eval.max_rows": 2, "eval.rows_per_batch": 2, "train.log_every": 1, "train.grad_probe_every": 1,
        "train.throughput_warmup_steps": 1, "semantic.min_chunks": 4}
SEM = ("lambda", "tau_s", "tau_r")


@pytest.fixture
def runs_dir(monkeypatch):
    d = ROOT / ".tmp" / "test_runs"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MOE_RUNS_DIR", str(d))
    monkeypatch.setenv("MOE_RESEARCH_APPROVE", "cpu-model")
    monkeypatch.chdir(ROOT)
    return d


def smoke_label():
    return json.loads(SMOKE_BASE.read_text())["result_label"]


def _cfg(cond, run_name, extra=None):
    from csr.config import build_condition_config, load_json
    return build_condition_config(load_json(SMOKE_BASE), cond, {**TINY, **(extra or {})}, run_name=run_name)


def _sets(d):
    out = []
    for k, v in d.items():
        out += ["--set", f"{k}={json.dumps(v)}"]
    return out


def _newest(runs_dir, suffix):
    c = sorted(p for p in runs_dir.iterdir() if p.name.endswith(suffix))
    assert c, suffix
    return c[-1]


# ------------------------------------------------------------------ guard interface
@pytest.mark.parametrize("cond", ["C", "E"])
def test_ordinary_planning_fails_without_calibration(cond, runs_dir):
    from csr.guard import GuardError
    from train import plan, run
    cfg = _cfg(cond, f"pytest_guard_{cond}")
    assert cfg["semantic"]["enabled"] and all(cfg["semantic"][k] is None for k in SEM)
    with pytest.raises(GuardError, match="not calibrated"):
        plan(cfg)
    with pytest.raises(GuardError, match="not calibrated"):
        plan(cfg, require_semantic_calibration=True)
    for mode in ("train", "benchmark"):  # run() is the train and benchmark path; it fails before any run dir
        before = set(runs_dir.iterdir())
        with pytest.raises(GuardError, match="not calibrated"):
            run(cfg, "cpu", True, mode, max_steps=1)
        assert set(runs_dir.iterdir()) == before


def test_calibration_planning_allowed_without_values():
    from train import plan
    cfg = _cfg("E", "pytest_calplan")
    d, tr, steps, total = plan(cfg, require_semantic_calibration=False)
    assert len(steps) > 0 and total > 0
    assert all(cfg["semantic"][k] is None for k in SEM)  # plan does not invent values
    _, _, steps_cal, total_cal = plan({**cfg, "semantic": {**cfg["semantic"], "lambda": 0.1, "tau_s": 0.1,
                                                           "tau_r": 0.01}})
    assert (steps, total) == (steps_cal, total_cal)  # identical data plan; only the guard differs


def test_bypass_not_reachable_from_train_or_benchmark():
    import train
    from train import plan, run
    p = inspect.signature(plan).parameters["require_semantic_calibration"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True
    assert "require_semantic_calibration" not in inspect.signature(run).parameters
    users = {}
    for f in ROOT.glob("*.py"):
        n = len(re.findall(r"\bplan\([^)]*require_semantic_calibration\s*=\s*False", f.read_text()))
        if n:
            users[f.name] = n
    # only calibration entry points (they produce the values); rep_calibrate_tau.py added by csr-representative-v1
    assert users == {"calibrate.py": 1, "rep_calibrate_tau.py": 1}, users
    assert re.search(r"plan\(cfg\)", inspect.getsource(train.run))  # run() uses the default
    for script in ("train.py", "benchmark.py"):  # no CLI flag can request the bypass
        r = subprocess.run([sys.executable, script, "--base", str(SMOKE_BASE), "--condition", "E",
                            "--require-semantic-calibration=False"], cwd=ROOT, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode != 0 and "unrecognized arguments" in r.stderr
        r = subprocess.run([sys.executable, script, "--base", str(SMOKE_BASE), "--condition", "E", "--device", "cpu"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "not calibrated" in r.stderr, r.stderr[-2000:]


# ------------------------------------------------------------------ calibration end to end
def test_calibrate_end_to_end_tiny_cpu_with_label(runs_dir):
    from calibrate import TAU_R_GRID, TAU_S_GRID
    r = subprocess.run([sys.executable, "calibrate.py", "--base", str(SMOKE_BASE), "--device", "cpu", "--batches", "1",
                        *_sets(TINY), "--execute"], cwd=ROOT, env=dict(os.environ), capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rd = _newest(runs_dir, "_calibrate")
    st = json.loads((rd / "summary.json").read_text())
    cfg = json.loads((rd / "config.json").read_text())
    assert st["status"] == "COMPLETE", st
    assert cfg["condition"] == "E" and all(cfg["semantic"][k] is None for k in SEM)  # no placeholder values
    assert st["tau_s"] in TAU_S_GRID and st["tau_r"] in TAU_R_GRID
    assert math.isfinite(st["lambda"]) and st["lambda"] > 0 and math.isfinite(st["weighted_ratio_at_lambda"])
    assert st["result_label"] == cfg["result_label"] == smoke_label()
    assert json.loads((rd / "status.json").read_text())["status"] == "COMPLETE"


# ------------------------------------------------------------------ evaluate / probe / train provenance
def test_train_evaluate_probe_carry_resolved_label(runs_dir):
    from train import run
    cfg = _cfg("A", "pytest_label_A")
    cfg["train"]["save_checkpoint"] = True
    label = cfg["result_label"]
    assert label == smoke_label()
    rd = run(cfg, "cpu", True, "train", max_steps=2)
    for f in ("summary.json", "run_facts.json", "config.json", "final_eval_docs.json"):
        assert json.loads((rd / f).read_text())["result_label"] == label, f
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "evaluate.py", "--run", str(rd), "--device", "cpu", "--max-rows", "2",
                        "--execute"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "COMPLETE" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    evs = sorted(rd.glob("eval_val_*.json"))
    assert len(evs) == 1
    ev = json.loads(evs[0].read_text())
    assert ev["status"] == "COMPLETE" and ev["result_label"] == label
    r = subprocess.run([sys.executable, "probe.py", "--checkpoint", str(rd / "model.pt"), "--device", "cpu",
                        "--n-receivers", "4", "--execute"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=600)
    assert "STATUS COMPLETE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    pd = _newest(runs_dir, "_probe")
    ps = json.loads((pd / "summary.json").read_text())
    pc = json.loads((pd / "config.json").read_text())
    assert ps["status"] == "COMPLETE" and ps["result_label"] == label
    assert pc["result_label"] == label == pc["model_cfg"]["result_label"]
