"""CPU integration of the real entry points on the real data with a TINY model (2 optimizer steps, a few rows).
Pipeline check only: no scientific claim. Writes under .tmp/test_runs. Skips if configs/v1 is absent."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "configs" / "v2" / "base.json"
pytestmark = pytest.mark.skipif(not BASE.exists(), reason="configs/v2 not generated")

TINY = {"model.d_model": 32, "model.n_head": 4, "model.dense_ffn": 48, "model.n_layer": 3, "model.moe_layers": [1, 2],
        "model.n_experts": 8, "model.top_k": 2, "model.expert_ffn": 16, "model.shared_ffn": 24,
        "routing.pool_size": 4, "train.micro_batch_rows": 2, "train.token_budget": 3000, "train.warmup_steps": 1,
        "eval.max_rows": 2, "eval.rows_per_batch": 2, "train.log_every": 1, "train.grad_probe_every": 1,
        "train.throughput_warmup_steps": 1, "semantic.lambda": 0.1, "semantic.tau_s": 0.1, "semantic.tau_r": 0.01,
        "semantic.min_chunks": 4}


@pytest.fixture
def runs_dir(monkeypatch):
    d = ROOT / ".tmp" / "test_runs"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MOE_RUNS_DIR", str(d))
    monkeypatch.setenv("MOE_RESEARCH_APPROVE", "cpu-model")
    monkeypatch.chdir(ROOT)
    return d


def _cfg(cond):
    from csr.config import build_condition_config, load_json
    return build_condition_config(load_json(BASE), cond, TINY, run_name=f"pytest_{cond}")


def test_train_all_conditions_tiny_cpu_matched(runs_dir):
    from train import run
    from csr.config import check_matched
    paths, facts, cfgs = {}, {}, {}
    for c in "ABCDEF":
        cfgs[c] = _cfg(c)
        cfgs[c]["train"]["save_checkpoint"] = c == "A"
        paths[c] = run(cfgs[c], "cpu", True, "train", max_steps=2)
        st = json.loads((paths[c] / "summary.json").read_text())
        if c == "F" and not cfgs[c]["segmentation"].get("random_cache"):
            assert st["status"] == "ABORTED" and "superseded" in st["reason"], st  # fail closed on F-v1
            continue
        assert st["status"] == "COMPLETE", (c, st)
        rf = json.loads((paths[c] / "run_facts.json").read_text())
        facts[c] = {"trainable_sha256": rf["trainable_sha256"], "valid_tokens": st["valid_train_tokens"],
                    "eval_tokens": st["final_eval"]["n_valid_tokens"]}
        m = [json.loads(l) for l in (paths[c] / "metrics.jsonl").read_text().splitlines()]
        steps = [r for r in m if "step" in r]
        cov = steps[-1]["coverage"]
        assert (cov == 0) == (c == "A")
        if c in "CE":
            assert steps[0]["router_gradnorm_sem_weighted"] > 0 and "sem_stats" in steps[0]
    norm = {c: {**cfg, "train": {**cfg["train"], "save_checkpoint": None}} for c, cfg in cfgs.items() if c in facts}
    assert check_matched(norm, facts)
    pytest.tiny_ckpt = paths["A"] / "model.pt"


def test_probe_and_evaluate_cpu(runs_dir):
    ck = getattr(pytest, "tiny_ckpt", None)
    if ck is None:
        pytest.skip("needs the training test")
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "probe.py", "--checkpoint", str(ck), "--device", "cpu", "--n-receivers", "6",
                        "--execute"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert "STATUS COMPLETE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    r = subprocess.run([sys.executable, "evaluate.py", "--run", str(ck.parent), "--device", "cpu", "--max-rows", "4",
                        "--execute"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.stdout.startswith("evaluate") and "COMPLETE" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([sys.executable, "evaluate.py", "--run", str(ck.parent), "--split", "test", "--device", "cpu",
                        "--execute"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "test split locked" in r.stderr
