"""ccsr1 frozen router grid (prereg section 5) and engineering-generation changes (section 9)."""
import json
from pathlib import Path

BASE = {"c": 1.0, "eta": 1.0, "H": 64, "lam": 0.01}
# 12 primary R3 configurations: base point, 6 one-factor changes, 5 fractional (c, eta) / (eta, lam) combinations
R3_GRID = [("bp", {}), ("c0.5", {"c": 0.5}), ("c2", {"c": 2.0}), ("e0.5", {"eta": 0.5}), ("e2", {"eta": 2.0}),
           ("h32", {"H": 32}), ("l0.1", {"lam": 0.1}), ("c0.5e2", {"c": 0.5, "eta": 2.0}), ("c2e0.5", {"c": 2.0, "eta": 0.5}),
           ("c2e2", {"c": 2.0, "eta": 2.0}), ("c0.5e0.5", {"c": 0.5, "eta": 0.5}), ("e2l0.1", {"eta": 2.0, "lam": 0.1})]
# controls at the base point (H 64, lambda 0.01): R1 semantic teacher; R2 utility teacher at each c
CONTROLS = [("R1", {"target": "R1"}), ("R2_c0.5", {"target": "R2", "c": 0.5}), ("R2_c1", {"target": "R2", "c": 1.0}),
            ("R2_c2", {"target": "R2", "c": 2.0})]
# engineering generations (cumulative, applied to every configuration)
GEN_CHANGES = {"E1": {"beta_eu": 1.0}, "E2": {"feats": "ctx"}, "E3": {"feats": "chg"}, "E4": {"n_out": 17},
               "E5": {"lam_mult": 10.0}}


def gen_changes(gen):
    """Changes of generations 1..gen, read from the logged generation records (ART/generations.json)."""
    from csr.ccsr1_paths import ART
    p = ART / "generations.json"
    if gen == 0:
        return []
    recs = json.loads(p.read_text())
    return [recs[str(g)]["trigger"] for g in range(1, gen + 1)]


def apply(cfg, triggers):
    c = dict(cfg)
    for t in triggers:
        ch = dict(GEN_CHANGES[t])
        if "lam_mult" in ch:
            c["lam"] = c["lam"] * ch.pop("lam_mult")
        if "beta_eu" in ch and c["target"] == "R1":
            ch.pop("beta_eu")       # R1 is the pure semantic control: no utility term
        c.update(ch)
    return c


def stage1(gen=0):
    tr = gen_changes(gen)
    out = []
    for tag, hp in R3_GRID:
        out.append(apply({"tag": f"R3_{tag}", "target": "R3", **BASE, **hp}, tr))
    for tag, hp in CONTROLS:
        c = {"tag": tag, **BASE, **hp}
        if c["target"] == "R1":
            c.pop("c")
            c.pop("eta")
        else:
            c.pop("eta")
        out.append(apply(c, tr))
    return out


if __name__ == "__main__":
    print(json.dumps(stage1(), indent=1))
    print(len(stage1()))
