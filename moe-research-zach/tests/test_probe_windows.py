"""probe.py receiver windows (no synthetic tokens or boundaries beyond a document), evaluated positions identical
across arms, no cross-document targets, nonzero exit on a non-COMPLETE probe, and top-level result_label.
CPU only (CUDA hidden), TINY untrained model, writes under .tmp/test_runs. Pipeline checks, no scientific claim."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SMOKE_BASE = ROOT / "configs" / "smoke-small-v1" / "base.json"
pytestmark = pytest.mark.skipif(not SMOKE_BASE.exists(), reason="smoke config not generated")

from probe import ARMS, doc_rows_batch, eval_targets, exit_code, receiver_window  # noqa: E402

TINY = {"model.d_model": 32, "model.n_head": 4, "model.dense_ffn": 48, "model.n_layer": 3, "model.moe_layers": [1, 2],
        "model.n_experts": 8, "model.top_k": 2, "model.expert_ffn": 16, "model.shared_ffn": 24,
        "routing.pool_size": 4}
W = 16


def _batch(doc, s, ln, n_rows=1):
    end, st, lo, hi = receiver_window(len(doc), s, ln, W)
    return doc_rows_batch([(doc[:end], st)] * n_rows, torch.device("cpu")), (end, st, lo, hi)


# ------------------------------------------------------------------ 1-3, 5: the window helper
def test_nonfinal_receiver_gets_following_token_and_boundary():
    doc = np.arange(100, 400)  # L = 300; receiver [64, 192) is not final
    tb, (end, st, lo, hi) = _batch(doc, 64, 128)
    assert end == 193 and st == [0, 64, 192] and (lo, hi) == (64 + W, 192)
    assert tb["input_ids"].shape[1] == 193 and int(tb["input_ids"][0, -1]) == doc[192]  # real following token
    assert int(tb["chunk_ids"][0, 192]) == 2 and int(tb["chunk_pos"][0, 192]) == 0  # real next chunk start
    pos, tg = eval_targets(tb, lo, hi)
    assert len(pos) == 128 - W and int(tg[0, -1]) == doc[192]  # last receiver token predicts the real next token
    # first chunk of the document (s = 0), non-final
    end, st, lo, hi = receiver_window(300, 0, 100, W)
    assert (end, st, lo, hi) == (101, [0, 100], W, 100)


def test_document_final_receiver_has_no_out_of_range_boundary():
    doc = np.arange(1000, 1177)  # the previously failing shape: a single 177-token chunk spanning the document
    end, st, lo, hi = receiver_window(177, 0, 177, W)
    assert end == 177 and st == [0] and (lo, hi) == (W, 176)
    assert all(x < end for x in st)
    tb, _ = _batch(doc, 0, 177)  # previously: AssertionError (array([0, 177]), 177) in starts_to_arrays
    assert tb["input_ids"].shape[1] == 177 and int(tb["chunk_ids"][0].max()) == 0
    # final chunk that starts mid-document
    end, st, lo, hi = receiver_window(300, 200, 100, W)
    assert end == 300 and st == [0, 200] and (lo, hi) == (200 + W, 299)
    for bad in ((300, 200, 101), (300, 300, 1), (300, 0, 10)):  # beyond the document / empty / shorter than w
        with pytest.raises(ValueError):
            receiver_window(bad[0], bad[1], bad[2], W)


def test_final_document_token_excluded_from_evaluation():
    doc = np.arange(5, 182)
    tb, (end, st, lo, hi) = _batch(doc, 0, 177)
    assert int(tb["targets"][0, 176]) == -100  # the last document token has no next-token target ...
    pos, tg = eval_targets(tb, lo, hi)
    assert int(pos[-1]) == 175 and 176 not in pos.tolist()  # ... and is not evaluated
    assert len(pos) == 177 - W - 1 and bool((tg != -100).all())


def test_no_target_crosses_a_document_boundary():
    for L, s, ln in ((300, 64, 128), (300, 200, 100), (177, 0, 177), (49, 0, 49)):
        doc = np.arange(L) + 7
        tb, (end, st, lo, hi) = _batch(doc, s, ln)
        assert end <= L and hi <= L - 1  # every target index hi-1+1 <= L-1 is a token of this document
        pos, tg = eval_targets(tb, lo, hi)
        assert bool((tg[0] == torch.as_tensor(doc[pos.numpy() + 1])).all())
    # a deliberately corrupted batch (target beyond the document / padding) is rejected
    tb, (end, st, lo, hi) = _batch(np.arange(177), 0, 177)
    with pytest.raises(AssertionError):
        eval_targets(tb, lo, hi + 1)


# ------------------------------------------------------------------ 4: arms
def test_evaluated_positions_identical_across_arms():
    doc = np.arange(300) + 3
    for s, ln in ((64, 128), (200, 100)):
        tb, (end, st, lo, hi) = _batch(doc, s, ln, n_rows=len(ARMS))
        pos, tg = eval_targets(tb, lo, hi)
        assert tg.shape == (len(ARMS), hi - lo) and all(torch.equal(tg[0], tg[b]) for b in range(len(ARMS)))
        tb["targets"][3, pos[0]] = 0 if int(tb["targets"][3, pos[0]]) else 1  # arms disagreeing -> rejected
        with pytest.raises(AssertionError):
            eval_targets(tb, lo, hi)


# ------------------------------------------------------------------ 6-8: end-to-end CPU probe
@pytest.fixture
def runs_dir(monkeypatch):
    d = ROOT / ".tmp" / "test_runs"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MOE_RUNS_DIR", str(d))
    monkeypatch.setenv("MOE_RESEARCH_APPROVE", "cpu-model")
    monkeypatch.chdir(ROOT)
    return d


def _tiny_checkpoint(runs_dir, name, corrupt=False):
    from csr.config import build_condition_config, load_json
    from csr.guard import utc_stamp
    from csr.train_lib import build_model
    cfg = build_condition_config(load_json(SMOKE_BASE), "A", TINY, run_name=name)
    torch.manual_seed(0)
    sd = {} if corrupt else build_model(cfg, torch.device("cpu")).state_dict()
    d = runs_dir / f"{utc_stamp()}_{name}"
    d.mkdir()
    torch.save({"model": sd, "config": cfg}, d / "model.pt")
    return d / "model.pt", cfg


def _newest_probe(runs_dir, before):
    new = sorted(p for p in set(runs_dir.iterdir()) - before if p.name.endswith("_probe"))
    assert len(new) == 1, new
    return new[0]


def test_probe_cpu_includes_document_final_receivers_with_label(runs_dir):
    ck, cfg = _tiny_checkpoint(runs_dir, "pytest_probe_ckpt")
    before = set(runs_dir.iterdir())
    r = subprocess.run([sys.executable, "probe.py", "--checkpoint", str(ck), "--device", "cpu", "--n-receivers", "12",
                        "--execute"], cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "STATUS COMPLETE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    pd = _newest_probe(runs_dir, before)
    rows = json.loads((pd / "probe_rows.json").read_text())
    assert len(rows) == 12
    final = [x for x in rows if x["doc_final"]]
    assert final, "receiver set must include a document-final chunk (the previously failing case)"
    assert any(x["doc"] == 7 and x["start"] == 0 and x["len"] == 177 == x["doc_len"] for x in final)
    for x in rows:
        e = x["start"] + x["len"]
        assert x["eval_lo"] == x["start"] + W and x["eval_hi"] == min(e, x["doc_len"] - 1)
        assert x["n_pos"] == x["eval_hi"] - x["eval_lo"] >= 32  # --min-eval-positions default
        assert x["n_pos"] == x["len"] - W - (1 if x["doc_final"] else 0)
        assert all(np.isfinite(x[f"nll_{a}"]) for a in ARMS)
    s = json.loads((pd / "summary.json").read_text())
    c = json.loads((pd / "config.json").read_text())
    st = json.loads((pd / "status.json").read_text())
    assert s["status"] == st["status"] == "COMPLETE" and s["n_receivers"] == 12
    assert s["n_positions"] == sum(x["n_pos"] for x in rows)
    assert s["result_label"] == c["result_label"] == cfg["result_label"] == json.loads(SMOKE_BASE.read_text())["result_label"]


def test_probe_failed_exits_nonzero_and_preserves_records(runs_dir):
    ck, _ = _tiny_checkpoint(runs_dir, "pytest_probe_badckpt", corrupt=True)  # empty state_dict -> load fails
    before = set(runs_dir.iterdir())
    r = subprocess.run([sys.executable, "probe.py", "--checkpoint", str(ck), "--device", "cpu", "--n-receivers", "2",
                        "--execute"], cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "STATUS FAILED" in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    pd = _newest_probe(runs_dir, before)
    s = json.loads((pd / "summary.json").read_text())
    assert s["status"] == json.loads((pd / "status.json").read_text())["status"] == "FAILED"
    assert "Traceback" in (pd / "log.txt").read_text()
    assert s["result_label"].startswith("ENGINEERING SMOKE TEST ONLY")


def test_exit_code_mapping():
    assert exit_code("COMPLETE") == 0 and exit_code(None) == 0  # None = dry-run
    assert all(exit_code(x) != 0 for x in ("FAILED", "ABORTED", "CONTAMINATED", "INCOMPLETE"))
