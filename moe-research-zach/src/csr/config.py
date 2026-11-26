"""Experiment configuration, conditions A-F, and the fail-closed constant-factor manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .guard import ROOT, sha256_json

CONDITIONS = {
    "A": {"segmentation.strategy": "none", "routing.mode": "token", "semantic.enabled": False},
    "B": {"segmentation.strategy": "fixed", "routing.mode": "pool", "semantic.enabled": False},
    "C": {"segmentation.strategy": "fixed", "routing.mode": "pool", "semantic.enabled": True},
    "D": {"segmentation.strategy": "semantic", "routing.mode": "pool", "semantic.enabled": False},
    "E": {"segmentation.strategy": "semantic", "routing.mode": "pool", "semantic.enabled": True},
    "F": {"segmentation.strategy": "random", "routing.mode": "pool", "semantic.enabled": False},
}
# The only fields allowed to differ between matched conditions.
APPROVED_DIFF = {"condition", "run_name", "segmentation.strategy", "routing.mode", "semantic.enabled"}
CONTRASTS = {
    "primary_H2": ("D", "F"), "primary_H3": ("E", "D"),
    "support_pool": ("B", "A"), "support_semloss_fixed": ("C", "B"), "support_boundary_with_semloss": ("E", "C"),
    "secondary_sem_vs_fixed": ("D", "B"),
}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def set_path(d, dotted, value):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


def load_json(p):
    return json.loads(Path(p).read_text())


def build_condition_config(base: dict, condition: str, overrides: dict | None = None, run_name=None):
    cfg = copy.deepcopy(base)
    for k, v in CONDITIONS[condition].items():
        set_path(cfg, k, v)
    for k, v in (overrides or {}).items():
        set_path(cfg, k, v)
    cfg["condition"] = condition
    cfg["run_name"] = run_name or f"{cfg.get('experiment', 'csr')}_{condition}"
    return cfg


def check_matched(configs: dict, extra_runtime: dict | None = None):
    """configs: {cond: cfg}. extra_runtime: {cond: {field: value}} of measured run facts (data hashes, trainable
    parameter names hash, valid-token counts, ...) that must also match. Raises ValueError listing every
    unapproved difference (fail closed)."""
    flats = {c: flatten(cfg) for c, cfg in configs.items()}
    if extra_runtime:
        for c, ex in extra_runtime.items():
            flats[c].update({f"runtime.{k}": v for k, v in ex.items()})
    keys = set().union(*[f.keys() for f in flats.values()])
    bad = []
    for k in sorted(keys):
        if k in APPROVED_DIFF:
            continue
        vals = {c: json.dumps(f.get(k, "<MISSING>"), sort_keys=True) for c, f in flats.items()}
        if len(set(vals.values())) > 1:
            bad.append((k, vals))
    if bad:
        msg = "; ".join(f"{k}: {v}" for k, v in bad)
        raise ValueError(f"manifest mismatch in unapproved fields: {msg}")
    return True


def manifest(configs: dict, data_manifest: dict | None, seg_manifest: dict | None, notes: dict | None = None):
    check_matched(configs)
    any_cfg = next(iter(configs.values()))
    shared = {k: v for k, v in flatten(any_cfg).items() if k not in APPROVED_DIFF}
    return {
        "schema": "csr-manifest-v1",
        "conditions": {c: {k: flatten(cfg)[k] for k in sorted(APPROVED_DIFF)} for c, cfg in configs.items()},
        "approved_differing_fields": sorted(APPROVED_DIFF),
        "shared_fields": shared,
        "shared_fields_sha256": sha256_json(shared),
        "contrasts": CONTRASTS,
        "data_manifest": data_manifest,
        "segmentation_manifest": seg_manifest,
        "notes": notes or {},
    }


BASE_CONFIG_PATH = ROOT / "configs" / "base.json"
