#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 Phase 2 (prereg sections 6-8). CALIB-SEARCH screening for K in {4, 8, 16, 32}
(semantic vs matched shuffled, seeds 0/1, 50M), repeatable-advantage rule, conditional engineering generations
(GEN-A / GEN-B, hashed docs), freeze, CALIB-SELECTION once, then the Phase-2B gate. Never reads VAL or domain labels.

  slk_phase2.py screen | decide | freeze | gate | all"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART, C6_BETA, GPU_CAP_H, MILESTONES  # noqa: E402

SEEDS = (0, 1)
KS = (4, 8, 16, 32)
S50 = MILESTONES["s50"]
P2 = ART / "phase2"


def R():
    import slk_run
    return slk_run


def tables():
    return json.loads((ART / "latent" / "tables.json").read_text())


def kmetrics():
    return json.loads((ART / "latent" / "k_metrics.json").read_text())["metrics"]


def p2a_hours():
    return R().gpu_hours({"search"}) + sum(1 for _ in [])  # search training; evaluations tracked in total


def p2a_used_h():
    """Phase-2A GPU hours actually used (training + evaluation), from RUN_LOG phase == phase2a (prereg section 10)."""
    tot = 0.0
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") == "phase2a" and r.get("job") != "slk_cluster.py all":
            tot += float(r.get("gpu_seconds") or 0)
    return tot / 3600


def p2a_job_means():
    """Mean GPU hours of completed Phase-2A screening training / dev-evaluation jobs (for generation projections)."""
    tr, ev = [], []
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") == "phase2a" and r.get("exit_status") == "COMPLETE" and r.get("gpu_seconds"):
            (tr if r.get("kind") == "train" else ev if r.get("kind") == "eval" else []).append(r["gpu_seconds"] / 3600)
    return (sum(tr) / len(tr) if tr else 0.25), (sum(ev) / len(ev) if ev else 0.25)


def gen_fits(n_cands, name):
    """Budget rule (user scope instruction 2026-11-16: GEN follow-ups only within the existing budget; prereg cap
    2A <= 12 GPU-h). Projection = n_cands x 4 runs x (train + 2 dev evals) + 4 CALIB-SELECTION evals."""
    bp = P2 / f"budget_{name.split()[0]}.json"
    if bp.exists():  # decided once; a resumed generation is never re-judged
        return json.loads(bp.read_text())["ok"]
    tr, ev = p2a_job_means()
    proj = n_cands * 4 * (tr + 2 * ev) + 4 * ev
    used = p2a_used_h()
    ok = used + proj <= GPU_CAP_H["phase2a"]
    guard.atomic_write_json(bp, {"time": guard.now(), "used_h": used, "projected_h": proj, "train_h": tr, "eval_h": ev,
                                 "cap_h": GPU_CAP_H["phase2a"], "ok": ok}, mode=0o400)
    R().decision(f"{name}: budget check", f"Phase-2A used {used:.2f} GPU-h; projected {proj:.2f} GPU-h "
                 f"(train {tr:.3f} h, dev eval {ev:.3f} h per job); cap {GPU_CAP_H['phase2a']}", "-",
                 "run generation" if ok else "SKIP generation (budget); recorded as not run, not as a negative result")
    return ok


def cand_runs(K, kind, seed, tag=""):
    """Run name for a screening condition; K = 16 maps to the Phase-1 runs (identical config up to 50M)."""
    if K == 16 and not tag:
        return f"P1_{'C6' if kind == 'SEM' else 'SHUF'}_s{seed}"
    return f"A_K{K}{tag}_{kind}_s{seed}"


def train_screen(K, kind, seed, table, sha, beta=C6_BETA, tag=""):
    from slk_configs import make, paper_cond
    rn = cand_runs(K, kind, seed, tag)
    if rn.startswith("P1_"):
        return rn
    st = R().status_of(rn)
    if st and "s50" in (st.get("ckpts") or {}):
        return rn
    if R().gpu_hours() + 0.25 > GPU_CAP_H["total"]:
        raise SystemExit("total GPU cap reached")
    p, _ = make(rn, seed, paper_cond(table, sha, beta), "search", diag_every=100)
    mine = json.loads(R().RUNS.read_text()) if R().RUNS.exists() else {}
    prev = mine.get(rn)
    argv = ["slk_train.py", "--config", str(p), "--max-steps", str(S50)] + (["--run-dir", prev] if prev else [])
    R().job(f"train_{rn}_to{S50}", argv, "phase2a", "train", rn, seed, p, expect=(75, 0), disk_gb=4.0)
    st = R().status_of(rn)
    if "s50" not in (st.get("ckpts") or {}):
        raise SystemExit(f"{rn}: no s50 milestone")
    R().drop_slot(rn, "screening run complete at its 50M milestone (never continued)")
    R().results(f"{rn} | train 50M screening | tokens {st['seen']}; steps {st['steps_done']} |")
    return rn


def dev(rn, tag, pseudo_cs, key=None):
    key = key or rn
    outp = ART / "evals" / key / f"dev_{tag}.json"
    if not outp.exists():
        R().job(f"dev_{key}_{tag}", ["slk_evaluate.py", "dev", "--run-dir", R().run_dirs()[rn], "--run-name", key,
                                     "--tag", tag, "--pseudo-table", pseudo_cs], "phase2a", "eval", key, None,
                disk_gb=0.5)
    e = json.loads(outp.read_text())
    a = e["audit"]
    row = {"ce": e["calib_search_B"]["ce"], "fsi_e5": a["e5"]["fsi"], "fsi_pseudo": a["pseudo"]["fsi"],
           "mi_e5": a["e5"]["routing"]["mi_mean"], "mi_pseudo": a["pseudo"]["routing"]["mi_mean"],
           "load_min_x": a["load_min_x_uniform"], "load_max_x": a["load_max_x_uniform"],
           "routing_entropy": a["routing_entropy"]}
    R().results(f"{key} | CALIB-SEARCH dev {tag} | CE {row['ce']:.4f}; FSI_e5 {row['fsi_e5']:.5f}; FSI_pseudo "
                f"{row['fsi_pseudo']:.5f}; MI_pseudo {row['mi_pseudo']:.4f} |")
    return row


def screen_K(K, tag="", T_tab=None, beta=C6_BETA):
    tb = tables()[f"K{K}"]
    sem, sem_sha = (T_tab or {}).get("sem", tb["sem"]), (T_tab or {}).get("sem_sha256", tb["sem_sha256"])
    shu, shu_sha = (T_tab or {}).get("shuf", tb["shuf"]), (T_tab or {}).get("shuf_sha256", tb["shuf_sha256"])
    out = {}
    for seed in SEEDS:
        for kind, t, s in (("SEM", sem, sem_sha), ("SHUF", shu, shu_sha)):
            rn = train_screen(K, kind, seed, t, s, beta, tag)
            key = f"A_K{K}{tag}_{kind}_s{seed}"
            for mt in ("s20", "s50"):
                out[f"{kind}_s{seed}_{mt}"] = dev(rn, mt, tb["calib_search"], key)
    return out


def summarize(K, rows, eligible):
    d = {}
    for seed in SEEDS:
        a, b = rows[f"SEM_s{seed}_s50"], rows[f"SHUF_s{seed}_s50"]
        d[seed] = {"d_e5": a["fsi_e5"] - b["fsi_e5"], "d_ps": a["fsi_pseudo"] - b["fsi_pseudo"],
                   "d_ce": a["ce"] - b["ce"], "pathology": bool(a["load_min_x"] < 0.1 or a["load_max_x"] > 4.0 or
                                                                a["routing_entropy"] < 0.45)}
    mean = {k: sum(d[s][k] for s in SEEDS) / len(SEEDS) for k in ("d_e5", "d_ps", "d_ce")}
    rep = bool(eligible and all(d[s]["d_e5"] > 0 for s in SEEDS) and all(d[s]["d_ps"] > 0 for s in SEEDS)
               and mean["d_ce"] <= 0.005 and not any(d[s]["pathology"] for s in SEEDS))
    return {"K": K, "per_seed": d, "mean": mean, "clustering_eligible": eligible, "repeatable": rep,
            "e5_pos_both": all(d[s]["d_e5"] > 0 for s in SEEDS)}


def screen():
    P2.mkdir(exist_ok=True)
    km = kmetrics()
    res = {}
    for K in KS:
        rows = screen_K(K)
        res[f"K{K}"] = {"rows": rows, **summarize(K, rows, km[f"SEM_K{K}"]["eligible"])}
    p = P2 / "screen.json"
    if not p.exists():
        guard.atomic_write_json(p, {"time": guard.now(), "results": res}, mode=0o400)
    return json.loads(p.read_text())["results"]


def write_gen(n, name, failure, hypothesis, cands):
    doc = guard.ROOT / "docs" / f"SLK_GEN{n}.md"
    if not doc.exists():
        txt = (f"# SLK engineering generation {n} — {name}\n\nWritten {guard.now()} before any run of this generation.\n\n"
               f"Observed failure: {failure}\n\nCausal hypothesis: {hypothesis}\n\nChange family: {name} (prereg "
               f"section 8). Candidates (each with its unchanged matched shuffled control, seeds 0/1, 50M, "
               f"CALIB-SEARCH):\n\n" + "".join(f"- {json.dumps(c)}\n" for c in cands) +
               "\nAcceptance rule (fixed before running): the prereg section-6 repeatable-advantage rule.\n")
        doc.write_text(txt)
        doc.chmod(0o400)
        (doc.parent / f"SLK_GEN{n}.md.sha256").write_text(f"{guard.sha256_file(doc)}  {doc.name}\n")
    return guard.sha256_file(doc)


def gen_tables(K, T):
    import numpy as np
    from csr.gc_lib import paper_q
    from csr.slk_lib import sharpen, shuffle_table
    km = kmetrics()
    med = km[f"SEM_K{K}"]["medoid_seed"]
    from slk_cluster import src_dir
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    q = paper_q(np.load(src_dir(K, med) / "view_logits.npz")["main"].astype(np.float64), zm["nv"])
    qs = sharpen(q, T)
    out = {}
    for name, arr in (("sem", qs), ("shuf", shuffle_table(zm["opaque"], qs, 20261117))):
        p = ART / "latent" / f"table_{name.upper()}_K{K}_T{str(T).replace('.', '')}.npz"
        if not p.exists():
            np.savez(p, opaque=zm["opaque"], q=arr.astype(np.float32), K=np.int64(K))
        out[name], out[f"{name}_sha256"] = str(p.relative_to(guard.ROOT)), guard.sha256_file(p)
    return out


def decide():
    res = screen()
    p = P2 / "decision_2a.json"
    if p.exists():
        return json.loads(p.read_text())
    gens = []
    n_gen = 0
    pool = {k: v for k, v in res.items()}
    none_rep = not any(v["repeatable"] for v in res.values())
    partial = [v["K"] for v in res.values() if v["e5_pos_both"] and not v["repeatable"] and v["clustering_eligible"]]
    if partial:
        n_gen += 1
        K = max(partial, key=lambda k: res[f"K{k}"]["mean"]["d_e5"])
        cands = [{"K": K, "beta": b, "T": 0.5} for b in (0.0025, 0.01)]
        if not gen_fits(len(cands), "GEN-B strength"):
            gens.append({"gen": n_gen, "name": "GEN-B", "cands": cands, "skipped": "budget (Phase-2A cap)"})
            cands = []
    if partial and cands:
        sha = write_gen(n_gen, "GEN-B strength", f"K={K} has FSI_e5(sem) > FSI_e5(shuf) in both seeds but fails "
                        f"the pseudo-FSI or CE part: {json.dumps(res[f'K{K}']['per_seed'])}",
                        "loss strength mis-set for this K", cands)
        R().decision("Phase-2A screening: partial semantic advantage", json.dumps(res[f"K{K}"]["per_seed"]),
                     "strength mis-set", f"run GEN-B (docs/SLK_GEN{n_gen}.md sha {sha[:12]})",
                     prediction="at most one of the two strengths becomes repeatable")
        for c in cands:
            tag = f"b{str(c['beta']).replace('.', '')}"
            rows = screen_K(K, tag=tag, beta=c["beta"])
            s = summarize(K, rows, True)
            pool[f"K{K}{tag}"] = {"rows": rows, **s, "gen": n_gen, "cand": c}
            gens.append({"gen": n_gen, "cand": c, "repeatable": s["repeatable"], "summary": s})
    if none_rep:
        n_gen += 1
        top = sorted(res.values(), key=lambda v: -v["mean"]["d_e5"])[:2]
        cands = [{"K": v["K"], "beta": C6_BETA, "T": 0.25} for v in top]
        if not gen_fits(len(cands), "GEN-A temperature"):
            gens.append({"gen": n_gen, "name": "GEN-A", "cands": cands, "skipped": "budget (Phase-2A cap)"})
            cands = []
    if none_rep and cands:
        sha = write_gen(n_gen, "GEN-A temperature", "no K has a repeatable semantic advantage at T = 0.5: " +
                        json.dumps({k: {"mean": v["mean"], "per_seed": v["per_seed"]} for k, v in res.items()}),
                        "diffuse soft labels limit the semantic signal; sharper q (T = 0.25) carries more of it", cands)
        R().decision("Phase-2A screening: no repeatable K", json.dumps({k: v["mean"] for k, v in res.items()}),
                     "diffuse labels", f"run GEN-A (docs/SLK_GEN{n_gen}.md sha {sha[:12]})",
                     prediction="T = 0.25 does not produce a repeatable advantage (v1 evidence: semantic increment "
                                "below seed noise)")
        for c in cands:
            tt = gen_tables(c["K"], 0.25)
            tag = "T025"
            rows = screen_K(c["K"], tag=tag, T_tab=tt)
            s = summarize(c["K"], rows, True)
            pool[f"K{c['K']}{tag}"] = {"rows": rows, **s, "gen": n_gen, "cand": c, "tables": tt}
            gens.append({"gen": n_gen, "cand": c, "repeatable": s["repeatable"], "summary": s})
    rep = [k for k, v in pool.items() if v["repeatable"]]
    km = kmetrics()
    winner = None
    if rep:
        best = max(pool[k]["mean"]["d_e5"] for k in rep)
        tied = [k for k in rep if best - pool[k]["mean"]["d_e5"] <= 0.0002]
        winner = max(tied, key=lambda k: km[f"SEM_K{pool[k]['K']}"]["S"])
    out = {"time": guard.now(), "screen": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in pool.items()},
           "generations": gens, "repeatable": rep, "winner": winner}
    guard.atomic_write_json(p, out, mode=0o400)
    R().decision("Phase-2A decision", json.dumps({k: (v["mean"], v["repeatable"]) for k, v in pool.items()}),
                 "-", f"winner {winner}" if winner else "no repeatable K: adaptive K not supported by 2A (2A ends)")
    return out


def freeze():
    dec = decide()
    if not dec["winner"]:
        return {"confirmed": False, "winner": None}
    w = dec["winner"]
    fz = P2 / "frozen_2a.json"
    if not fz.exists():
        guard.atomic_write_json(fz, {"time": guard.now(), "winner": w, "summary": dec["screen"][w]}, mode=0o400)
    if not (ART / "CALIBSEL_OPEN.json").exists():
        guard.atomic_write_json(ART / "CALIBSEL_OPEN.json", {"time": guard.now(), "frozen": w}, mode=0o400)
    s = dec["screen"][w]
    K = s["K"]
    tag = w[len(f"K{K}"):]
    tb = tables()[f"K{K}"]
    rows = {}
    for seed in SEEDS:
        for kind in ("SEM", "SHUF"):
            rn = cand_runs(K, kind, seed, tag)
            key = f"A_K{K}{tag}_{kind}_s{seed}"
            outp = ART / "evals" / key / "calibsel_s50.json"
            if not outp.exists():
                R().job(f"calibsel_{key}", ["slk_evaluate.py", "calibsel", "--run-dir", R().run_dirs()[rn], "--run-name",
                                            key, "--tag", "s50", "--pseudo-table", tb["calib_search"],
                                            "--pseudo-table-sel", tb["calib_sel"]], "phase2a", "eval", key, seed,
                        disk_gb=0.5)
            e = json.loads(outp.read_text())
            rows[f"{kind}_s{seed}"] = {"ce": e["calib_sel"]["ce"], "fsi_e5": e["audit"]["e5"]["fsi"],
                                       "fsi_pseudo": e["audit"]["pseudo"]["fsi"]}
    d = {s_: {m: rows[f"SEM_s{s_}"][m] - rows[f"SHUF_s{s_}"][m] for m in ("ce", "fsi_e5", "fsi_pseudo")} for s_ in SEEDS}
    conf = bool(all(d[s_]["fsi_e5"] > 0 and d[s_]["fsi_pseudo"] > 0 for s_ in SEEDS)
                and sum(d[s_]["ce"] for s_ in SEEDS) / 2 <= 0.005)
    out = {"time": guard.now(), "winner": w, "rows": rows, "diff": d, "confirmed": conf}
    p = P2 / "calibsel_2a.json"
    if not p.exists():
        guard.atomic_write_json(p, out, mode=0o400)
    R().results(f"{w} | CALIB-SELECTION (once) | {json.dumps(d)}; confirmed {conf} |")
    R().decision("Phase-2A CALIB-SELECTION", json.dumps(d), "-", f"K* {'confirmed' if conf else 'NOT confirmed'}")
    return json.loads(p.read_text())


def gate():
    p1 = json.loads((ART / "phase1_decision.json").read_text())
    c2a = freeze()
    open_ = bool(p1["semantic_beyond_grouping"] or c2a.get("confirmed"))
    g = {"time": guard.now(), "phase1_R_and_F": p1["semantic_beyond_grouping"], "phase2a_confirmed": c2a.get("confirmed"),
         "gate_open": open_}
    p = P2 / "gate_2b.json"
    if not p.exists():
        guard.atomic_write_json(p, g, mode=0o400)
    R().decision("Phase-2B gate", json.dumps(g), "-", "run Phase 2B" if open_ else "Phase 2B not run (gate closed)")
    return json.loads(p.read_text())


BLOCK_2B = ART / "PHASE2B_BLOCKED_BY_USER.json"
STATE_STOP = "STOPPED_BEFORE_PHASE2B_BY_USER"


def stop_before_2b(g):
    """User scope reduction 2026-11-16 (DEVIATIONS n=2): no Phase-2B job is dispatched. Orchestration only."""
    st = {"time": guard.now(), "state": STATE_STOP, "gate_2b": g, "block_marker": str(BLOCK_2B.relative_to(guard.ROOT)),
          "phase2b_dispatched": False}
    p = ART / "STATE.json"
    guard.atomic_write_json(p, st, overwrite=True)
    R().decision("Stage boundary: Phase 2A complete; Phase 2B blocked by user scope reduction", json.dumps(g), "-",
                 f"do not dispatch any Phase-2B job; state {STATE_STOP}; the gate would have been "
                 f"{'OPEN' if g['gate_open'] else 'CLOSED'}")
    R().write_resume(
        f"{STATE_STOP} — Phase 1 and Phase 2A (with permitted GEN-A/GEN-B follow-ups) complete; no Phase-2B job was "
        f"dispatched (user instruction 2026-11-16). Preregistered gate would have been "
        f"{'OPEN' if g['gate_open'] else 'CLOSED'}. GPU released.",
        ".venv/bin/python slk_phase2.py all --allow-phase2b",
        "Do NOT run this without explicit user instruction. Prerequisites before it can work: `slk_phase2b.py` "
        "(prereg section 7: Ward hierarchy of K32 into 16/8/4, controller w = 0.05 + 0.8*softmax(a_l), lambda_c 0.05, "
        "beta 0.005; pilot, then ADAPT / ADAPT-SHUF / INERT / best-fixed-K finals, seeds 0-2) and a hierarchical "
        "condition in slk_train.py are NOT yet implemented. If the gate is CLOSED the preregistration does not run "
        "Phase 2B at all and the command only re-records the gate. Phase-1/2A steps are idempotent (finished jobs "
        "and frozen 0400 files are skipped).")
    return st


def run_all(allow_2b=False, stage_limit=False):
    import slk_analyze
    slk_analyze.phase1_decision()
    g = gate()
    if stage_limit or (BLOCK_2B.exists() and not allow_2b):
        stop_before_2b(g)
        return
    if g["gate_open"]:
        import slk_phase2b
        slk_phase2b.run_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "decide", "freeze", "gate", "all"])
    ap.add_argument("--allow-phase2b", action="store_true", help="override the user Phase-2B block (explicit "
                                                                 "user instruction only)")
    a = ap.parse_args()
    guard.check_root()
    from csr.slk_paths import check_prereg
    check_prereg()
    if a.cmd == "all":
        run_all(a.allow_phase2b)
    else:
        {"screen": screen, "decide": decide, "freeze": freeze, "gate": gate}[a.cmd]()


if __name__ == "__main__":
    main()
