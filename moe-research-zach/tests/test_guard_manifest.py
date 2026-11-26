"""Gates 14-15: interlocks, limits, atomic IO, manifests and stale caches."""
import os
import time

import pytest

from csr import guard
from csr.config import CONDITIONS, build_condition_config, check_matched
from csr.segcache import cache_id


def base_cfg():
    return {"experiment": "t", "model": {"d_model": 32}, "routing": {"mode": "token", "pool_size": 12, "warmup": 16},
            "segmentation": {"strategy": "none", "fixed_size": 128}, "semantic": {"enabled": False, "lambda": 0.1},
            "init": {"type": "checkpoint", "path": "x.pt", "sha256": "aa"}, "data": {"manifest_sha256": "d1"},
            "train": {"steps": 10, "seed": 1}}


def test_interlock(monkeypatch):
    monkeypatch.delenv("MOE_RESEARCH_APPROVE", raising=False)
    assert guard.require_execution(False, "gpu") is False
    with pytest.raises(guard.GuardError):
        guard.require_execution(True, "gpu")
    monkeypatch.setenv("MOE_RESEARCH_APPROVE", "download")
    with pytest.raises(guard.GuardError):
        guard.require_execution(True, "gpu")
    monkeypatch.setenv("MOE_RESEARCH_APPROVE", "gpu")
    assert guard.require_execution(True, "gpu") is True


def test_wall_clock_and_vram_limits(monkeypatch):
    lim = guard.RunLimits(max_seconds=0.05)
    time.sleep(0.1)
    with pytest.raises(guard.GuardError, match="wall-clock"):
        lim.check()
    lim.close()
    lim = guard.RunLimits(max_seconds=100)
    monkeypatch.setattr(lim, "peak_gib", lambda: (54.0, 54.0))
    with pytest.raises(guard.GuardError, match="VRAM"):
        lim.check()
    lim.close()


def test_contamination_flag_stops_run():
    class M:
        contaminated, detail = True, "foreign"
    lim = guard.RunLimits(max_seconds=100, monitor=M())
    with pytest.raises(guard.GuardError, match="CONTAMINATED"):
        lim.check()
    lim.close()


def test_occupancy_gate_blocks_when_gpu_busy(monkeypatch):
    monkeypatch.setattr(guard, "gpu_processes", lambda i=0: [12345])
    monkeypatch.setattr(guard, "ledger", lambda *a: None)
    with pytest.raises(guard.GuardError, match="BLOCKED"):
        guard.occupancy_gate("test")


def test_disk_reserve(monkeypatch):
    monkeypatch.setattr(guard, "ledger", lambda *a: None)
    with pytest.raises(guard.GuardError):
        guard.require_disk(10**15, "huge")


def test_atomic_write_refuses_overwrite_and_escape(tmp_path_factory):
    d = guard.ROOT / ".tmp" / f"pytest_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "a.json"
    guard.atomic_write_json(p, {"x": 1})
    with pytest.raises(guard.GuardError):
        guard.atomic_write_json(p, {"x": 2})
    with pytest.raises(guard.GuardError):
        guard.inside_root("/etc/passwd")


def test_manifest_accepts_only_approved_differences():
    cfgs = {c: build_condition_config(base_cfg(), c) for c in CONDITIONS}
    assert check_matched(cfgs)
    for field, val in [("init.path", "other.pt"), ("data.manifest_sha256", "d2"), ("train.steps", 11),
                       ("semantic.lambda", 0.2), ("routing.pool_size", 16), ("model.d_model", 64)]:
        bad = dict(cfgs)
        bad["E"] = build_condition_config(base_cfg(), "E", {field: val})
        with pytest.raises(ValueError, match=field.replace(".", r"\.")):
            check_matched(bad)


def test_manifest_runtime_facts_must_match():
    cfgs = {c: build_condition_config(base_cfg(), c) for c in ("A", "B")}
    with pytest.raises(ValueError, match="trainable_sha"):
        check_matched(cfgs, {"A": {"trainable_sha": "x", "valid_tokens": 5}, "B": {"trainable_sha": "y", "valid_tokens": 5}})
    with pytest.raises(ValueError, match="valid_tokens"):
        check_matched(cfgs, {"A": {"valid_tokens": 5}, "B": {"valid_tokens": 6}})


def test_cache_id_changes_with_any_input():
    spec = {"data": "abc", "tau": 0.3, "encoder": "rev1", "w": 16}
    ids = {cache_id(spec)}
    for k, v in [("data", "abd"), ("tau", 0.31), ("encoder", "rev2"), ("w", 8)]:
        ids.add(cache_id({**spec, k: v}))
    assert len(ids) == 5
