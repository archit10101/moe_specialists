#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1: discovery export (prereg section 2).

  gc_data.py export --split main|calib_search|calib_sel|val
Writes artifacts/gc-edl-v1/discovery/inputs_<split>.npz with ONLY: opaque paper IDs (sorted), number of views, view
token arrays (P, 4, 256) uint16 (zero padded), view starts in the concatenated paper stream, paper token length and
number of segments. No domain, domain name, path, paper hash or document ID is written. The opaque -> paper map goes
to artifacts/gc-edl-v1/sealed/opaque_map_<split>.json (trainer/evaluator only; discovery is firewalled from it).
val is refused until artifacts/gc-edl-v1/frozen.json exists. This program reads label-bearing split metadata only to
group segments into papers; it never writes or returns a label value.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ALLOWED_KEYS = {"opaque", "nv", "views", "starts", "paper_len", "n_seg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export"])
    ap.add_argument("--split", required=True, choices=["main", "calib_search", "calib_sel", "val"])
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    import numpy as np
    from csr.fsp_lib import Split
    from csr.gc_lib import opaque_id, view_starts
    from csr.gc_paths import ART, DISC, SEALED, VIEW, check_prereg
    check_prereg()
    if a.split == "val" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL export only after the freeze")
    out = DISC / f"inputs_{a.split}.npz"
    if out.exists():
        print(f"{out} exists")
        return
    sp = Split(a.split)
    segs = {}
    for di, (p, sid) in enumerate(zip(sp.papers, sp.ids)):
        segs.setdefault(p, []).append((int(sid.split("#")[1]), di))
    rows = []
    for p, lst in segs.items():
        lst.sort()
        assert [k for k, _ in lst] == list(range(len(lst))), "non-contiguous segments"
        rows.append((opaque_id(p), p, [di for _, di in lst]))
    rows.sort()
    P = len(rows)
    views = np.zeros((P, 4, VIEW), np.uint16)
    starts = np.full((P, 4), -1, np.int64)
    nv = np.zeros(P, np.int64)
    plen = np.zeros(P, np.int64)
    nseg = np.zeros(P, np.int64)
    omap = {}
    for i, (o, p, dis) in enumerate(rows):
        toks = np.concatenate([sp.doc(d) for d in dis])
        st = view_starts(len(toks), VIEW)
        nv[i], plen[i], nseg[i] = len(st), len(toks), len(dis)
        for j, s in enumerate(st):
            views[i, j] = toks[s:s + VIEW]
            starts[i, j] = s
        omap[o] = p
    assert len(set(omap)) == P
    rec = {"opaque": np.asarray([r[0] for r in rows]), "nv": nv, "views": views, "starts": starts,
           "paper_len": plen, "n_seg": nseg}
    assert set(rec) == ALLOWED_KEYS
    DISC.mkdir(parents=True, exist_ok=True)
    SEALED.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **rec)
    tmp.replace(out)
    guard.atomic_write_json(SEALED / f"opaque_map_{a.split}.json", omap, mode=0o400)
    sha = guard.sha256_file(out)
    with open(ART / "artifact_hashes.tsv", "a") as f:
        f.write(f"{guard.now()}\t{out.relative_to(guard.ROOT)}\t{sha}\t{out.stat().st_size}\n")
        f.write(f"{guard.now()}\tsealed/opaque_map_{a.split}.json\t"
                f"{guard.sha256_file(SEALED / f'opaque_map_{a.split}.json')}\t\n")
    print(json.dumps({"split": a.split, "papers": P, "views": int(nv.sum()),
                      "nv_hist": np.bincount(nv, minlength=5).tolist(), "sha256": sha}))


if __name__ == "__main__":
    main()
