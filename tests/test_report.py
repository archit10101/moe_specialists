"""The report must render with any subset of runs finished (e.g. the paper tiers, which never run §18)."""
import json
import os

import pytest

from repro.config import SMOKE


def _write(work, exp, seed, d):
    f = work / "smoke" / "runs" / exp / f"s{seed}" / "eval.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d))


def _report(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_WORK", str(tmp_path))
    monkeypatch.delenv("REPRO_REPORT_DIR", raising=False)
    import importlib
    import repro.config, repro.utils, repro.report
    importlib.reload(repro.config); importlib.reload(repro.utils); importlib.reload(repro.report)
    repro.report.write_report(repro.config.SMOKE)
    return tmp_path / "smoke" / "results"


def test_report_with_no_runs_at_all(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    rep, paper = (out / "REPORT.md").read_text(), (out / "PAPER.md").read_text()
    assert "§18" in rep and "**not run**" in rep and "failed to render" not in rep + paper
    assert "_not run yet_" in paper
    assert (out / "per_seed.csv").read_text().splitlines() == ["experiment,seed,metric,value"]


def test_report_with_paper_runs_only(tmp_path, monkeypatch):
    bank = lambda imp, spec=0.001: dict(improvement=imp, base_ce=3.0, bank_ce=3.0 - imp,
                                        swap_own=dict(specificity=spec, ci=[spec / 2, spec * 2], own_best_frac=1.0,
                                                      swap_matrix=[[3.0, 3.1], [3.1, 3.0]]))
    for s in range(4):
        _write(tmp_path, "bank.sem", s, bank(0.003 + 0.0001 * s))
        _write(tmp_path, "bank.shuf", s, bank(0.001, 0.0))
        _write(tmp_path, "bank.tok", s, bank(0.002, 0.0))
        for f, ce in ((10, 4.0), (25, 3.6), (50, 3.3)):
            for r, v in (("sem", 0.0005 * f / 10), ("shuf", 0.0002), ("tok", 0.0003)):
                d = bank(v); d["base_ce"] = ce
                _write(tmp_path, f"abl.bank_{r}_at{f}", s, d)
    for s in range(3):
        _write(tmp_path, "router.sem", s, dict(router_minus_g=-0.004, router_ce=3.0, oracle_ce=2.99,
                                               by_position_router_minus_g=dict(first32=0.002, after_boundary=0.001, stable=-0.007),
                                               position_token_share=dict(first32=0.03, after_boundary=0.06, stable=0.91)))
        _write(tmp_path, "router.shuf", s, dict(router_minus_g=0.0, router_ce=3.004, oracle_ce=2.995))
    out = _report(tmp_path, monkeypatch)
    rep, paper = (out / "REPORT.md").read_text(), (out / "PAPER.md").read_text()
    assert "failed to render" not in rep + paper
    assert rep.count("**not run**") >= 7                                # every §18/§21 claim
    assert "PASS" in rep                                                # §23 claims evaluated
    assert "4/4 > 0" in paper and "3/3 < 0" in paper                     # sign counts
    assert "## B4" in paper and "slope vs base val CE" in paper
    assert "## B8" not in paper                                         # extras only when present
    assert "Router by position" in paper and "## B6" in paper and "_not run yet_" in paper
    assert (out / "figures" / "swap_matrix.png").exists() and (out / "figures" / "b4_gap_vs_base.pdf").exists()
    rows = (out / "per_seed.csv").read_text().splitlines()
    assert "bank.sem,0,improvement,0.003" in rows
