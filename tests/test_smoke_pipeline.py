"""End-to-end: the whole core pipeline on the smoke profile (synthetic data, ~1-2 min on a laptop CPU).
Skipped unless REPRO_SLOW=1 (the cloud prompt and CI run it explicitly)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("REPRO_SLOW") != "1", reason="set REPRO_SLOW=1 to run")
def test_smoke_pipeline_end_to_end(tmp_path):
    # bf16 autocast on CPU, as on a GPU: dtype mismatches only show up under mixed precision
    env = dict(os.environ, REPRO_WORK=str(tmp_path), REPRO_DEVICE="cpu", REPRO_CPU_AUTOCAST="1")
    subprocess.run([sys.executable, "-m", "repro.run", "local", "--profile", "smoke"], cwd=ROOT, env=env,
                   check=True)
    res = tmp_path / "smoke" / "results"
    assert (res / "REPORT.md").exists()
    claims = json.loads((res / "summary.json").read_text())["claims"]
    assert all(c["verdict"] in ("PASS", "FAIL") for c in claims), "every claim should be evaluated"
    # the bank swap matrix covers all of the evaluated val tokens, not a small probe
    ev = json.loads((tmp_path / "smoke" / "runs" / "bank.sem" / "s0" / "eval.json").read_text())
    assert ev["swap_tokens"] == ev["val_tokens"] if "val_tokens" in ev else ev["swap_tokens"] > 0
    # resuming a finished pipeline must be a no-op
    out = subprocess.run([sys.executable, "-m", "repro.run", "local", "--profile", "smoke"], cwd=ROOT, env=env,
                         check=True, capture_output=True, text=True).stdout
    assert "=== base:" not in out
