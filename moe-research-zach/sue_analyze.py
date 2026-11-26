#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 analysis (CPU; reads frozen stage outputs and evaluation JSONs). Writes (derived,
rewritten): final_analysis.json, SPECIALIZATION_TRAJECTORY.json, run_manifest.tsv, tables.md."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.sue_paths import ART, GPU_CAP_H  # noqa: E402


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def compute():
    by_phase, by_kind = {}, {}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        h = float(r.get("gpu_seconds") or 0) / 3600
        by_phase[r.get("phase", "?")] = by_phase.get(r.get("phase", "?"), 0.0) + h
        k = f"{r.get('phase')}/{r.get('kind')}"
        by_kind[k] = by_kind.get(k, 0.0) + h
    led = 0.0
    if (ART / "compute_ledger.tsv").exists():
        for line in (ART / "compute_ledger.tsv").read_text().splitlines():
            try:
                led += float(line.split("\t")[2]) / 3600
            except (IndexError, ValueError):
                pass
    return {"by_phase": by_phase, "by_phase_kind": by_kind, "compute_ledger_total_h": led, "caps": GPU_CAP_H}


def trajectory():
    out = {}
    for d in sorted((ART / "evals").glob("*")) if (ART / "evals").exists() else []:
        for f in sorted(d.glob("*.json")):
            e = jl(f)
            pe = e["per_expert"]["pseudo"]
            out.setdefault(d.name, {})[f"{e['mode']}_{e['tag']}"] = {
                "step": e["ckpt"]["step"], "tokens": e["ckpt"]["seen"], "block": e["block"], "ce": e["ce_block"]["ce"],
                "pseudo_interaction": pe["mean_interaction"], "pseudo_interaction_ci95": pe["mean_interaction_ci95"],
                "n_specialized": pe["n_subject_specialized"], "pseudo_mi": e["audit"]["pseudo"]["routing"]["mi_mean"],
                "pseudo_nmi": e["audit"]["pseudo"]["routing"].get("nmi_mean"),
                "pseudo_topic_pred": e["audit"]["pseudo"].get("topic_pred_acc"),
                "pseudo_fsi": e["audit"]["pseudo"].get("fsi"), "e5_interaction": e["per_expert"]["e5"]["mean_interaction"],
                "generic_importance": sum(e["per_expert"]["generic_importance_mean_dce"]) /
                len(e["per_expert"]["generic_importance_mean_dce"]), **e["pathology"]}
    return out


def main():
    guard.check_root()
    import sue_run as R
    fa = {"time": guard.now(), "grouping": jl(ART / "grouping.json"), "diagnostic": jl(ART / "diagnostic.json"),
          "gamma": jl(ART / "gamma.json"), "search": jl(ART / "search_decision.json"), "promotion": jl(ART / "frozen.json"),
          "confsel": jl(ART / "confsel.json"), "final": jl(ART / "final_decision.json"), "compute": compute()}
    traces = {}
    for rn, rd in R.run_dirs().items():
        p = guard.ROOT / rd / "metrics.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            traces[rn] = {"util_diag": [x["util_diag"] for x in rows if "util_diag" in x][-1:] or None,
                          "use": [(x["step"], x.get("use"), x.get("use_margin"), x.get("use_active"), x.get("cf_s"),
                                   x.get("dt"), x.get("lm")) for x in rows if "step" in x and x["step"] % 25 == 0],
                          "status": R.status_of(rn)}
    fa["run_traces"] = traces
    fa["deviations"] = [json.loads(x) for x in (ART / "DEVIATIONS.jsonl").read_text().splitlines()] \
        if (ART / "DEVIATIONS.jsonl").exists() else []
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    fa["integrity"] = jl(inv[-1]) if inv else None
    sz = lambda p: int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0])  # noqa
    own = [rd for rd in R.run_dirs().values()]
    fa["disk"] = {"own_runs": len(own), "own_runs_gib": sum(sz(guard.ROOT / d) for d in own) / 2**30,
                  "artifacts_gib": sz(ART) / 2**30, "free_gib": guard.disk_free_bytes() / 2**30}
    guard.atomic_write_json(ART / "final_analysis.json", fa, overwrite=True)
    guard.atomic_write_json(ART / "SPECIALIZATION_TRAJECTORY.json", trajectory(), overwrite=True)
    rows = ["run_name\trun_dir\tstatus\tseed\tsteps\ttokens\tconfig_sha256\tplan_sha256\tcond"]
    for rn, rd in sorted(R.run_dirs().items()):
        st = R.status_of(rn) or {}
        cfgp = guard.ROOT / rd / "config.json"
        cfg = jl(cfgp) or {}
        rows.append("\t".join(str(x) for x in (rn, rd, st.get("status"), cfg.get("seed"), st.get("steps_done"),
                                                st.get("seen"), guard.sha256_file(cfgp) if cfgp.exists() else None,
                                                st.get("plan_sha256"), json.dumps(cfg.get("cond")))))
    guard.atomic_write_bytes(ART / "run_manifest.tsv", ("\n".join(rows) + "\n").encode(), overwrite=True)
    print(json.dumps({"compute": fa["compute"]["by_phase"], "disk": fa["disk"]}))


if __name__ == "__main__":
    main()
