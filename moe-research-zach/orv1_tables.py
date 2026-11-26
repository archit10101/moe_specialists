#!/usr/bin/env python
"""oracle-residual-v1 Stage 0 (CPU): frozen label-free assignments, shuffled controls, training plan, evaluation
partitions and token features. Written once (0400) under artifacts/oracle-residual-v1/tables/. No domain label is read.

  labels_main.npz / labels_cs.npz : per-document hard semantic label (argmax of the frozen GC-EDL SEM K16 table) and the
                                    shuffled label (-1 = paper without semantic features)
  shuffled control                : the hard labels permuted across papers WITHIN paper-token-mass bins (20 quantile
                                    bins; fixed seeds) -> papers per cluster preserved exactly, token mass approximately
  plan.npz                        : residual-training stream = main documents NOT seen by the TOKEN-4 base plan (and with
                                    an assignment), in a fixed sha256 order, packed into 1024-token rows, 32 rows/step,
                                    sized for the largest budget (prefix plans for smaller budgets)
  parts.npz                       : SEL = CALIB-SEARCH paper-half A, EVAL = CALIB-SEARCH paper-half B (assigned docs)
  token_feats.npz                 : unigram frequency of every vocab id on the main split; target-token class
  tables.json                     : hashes and descriptive statistics"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def u_hash(tag, s):
    return int.from_bytes(hashlib.sha256(f"{tag}|{s}".encode()).digest()[:8], "little") / 2**64


def hard_labels(path, sha, split):
    import numpy as np
    from csr.gc_lib import opaque_id
    if guard.sha256_file(path) != sha:
        raise guard.GuardError(f"table hash mismatch {path}")
    z = np.load(path)
    idx = {str(o): i for i, o in enumerate(z["opaque"])}
    q = z["q"]
    lab = np.full(split.n_docs, -1, np.int64)
    cache = {}
    for di, p in enumerate(split.papers):
        if p not in cache:
            i = idx.get(opaque_id(p))
            cache[p] = int(q[i].argmax()) if (i is not None and q[i].sum() > 0) else -1
        lab[di] = cache[p]
    return lab


def length_stratified_shuffle(split, lab, seed, nbins):
    """Permute paper labels within paper-token-mass quantile bins. Returns doc-level shuffled labels and stats."""
    import numpy as np
    from csr.gc_lib import opaque_id
    papers = sorted({p for p, l_ in zip(split.papers, lab) if l_ >= 0}, key=opaque_id)
    pidx = {p: i for i, p in enumerate(papers)}
    mass = np.zeros(len(papers))
    plab = np.full(len(papers), -1)
    for di, p in enumerate(split.papers):
        if p in pidx:
            mass[pidx[p]] += split.lengths[di] - 1
            plab[pidx[p]] = lab[di]
    cuts = np.quantile(mass, np.linspace(0, 1, nbins + 1)[1:-1])
    b = np.searchsorted(cuts, mass, side="right")
    rng = np.random.default_rng(seed)
    sh = plab.copy()
    for k in range(nbins):
        ii = np.nonzero(b == k)[0]
        sh[ii] = plab[ii[rng.permutation(len(ii))]]
    out = np.full(split.n_docs, -1, np.int64)
    for di, p in enumerate(split.papers):
        if p in pidx:
            out[di] = sh[pidx[p]]
    return out, {"n_papers": len(papers), "papers_per_cluster_equal": bool(
        (np.bincount(plab, minlength=16) == np.bincount(sh, minlength=16)).all()),
        "hard_agreement": float((plab == sh).mean())}


def mass(split, docs, lab, K=16):
    import numpy as np
    w = split.lengths[docs] - 1
    m = np.bincount(lab[docs], weights=w, minlength=K)
    return (m / m.sum()).round(4).tolist()


def main():
    guard.check_root()
    import numpy as np
    from csr.fsp_lib import Split, pack_rows, plan_steps
    from csr.lfr_eval import paper_half
    from csr.lfr_lib import main_stream
    from csr.orv1_paths import (BUDGETS, K, ROWS_PER_STEP, SEM_CS, SEM_CS_SHA, SEM_MAIN, SEM_MAIN_SHA, SHUF_LEN_BINS,
                                SHUF_SEED_CS, SHUF_SEED_MAIN, T4_BUDGET, T4_PLAN_SHA, TABLES, TOKENIZER)
    TABLES.mkdir(parents=True, exist_ok=True)
    if (TABLES / "tables.json").exists():
        print("exists")
        return
    out = {"time": guard.now(), "sources": {"SEM_MAIN": [str(SEM_MAIN.relative_to(guard.ROOT)), SEM_MAIN_SHA],
                                            "SEM_CS": [str(SEM_CS.relative_to(guard.ROOT)), SEM_CS_SHA]}}
    sp = Split("main")
    _, steps, tot = main_stream(T4_BUDGET)
    psha = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    if psha != T4_PLAN_SHA:
        raise guard.GuardError("TOKEN-4 base plan hash mismatch")
    seen = np.zeros(sp.n_docs, bool)
    seen[[d for st in steps for r in st for d in r]] = True
    lab_m = hard_labels(SEM_MAIN, SEM_MAIN_SHA, sp)
    sh_m, st_m = length_stratified_shuffle(sp, lab_m, SHUF_SEED_MAIN, SHUF_LEN_BINS)
    pool = np.nonzero(~seen & (lab_m >= 0))[0]
    pool = sorted(pool.tolist(), key=lambda i: u_hash("orv1-order", f"{sp.papers[i]}|{i}"))
    rows = pack_rows(sp.lengths, pool)
    psteps, ptot = plan_steps(rows, sp.lengths, ROWS_PER_STEP, max(BUDGETS.values()))
    flat = [d for st in psteps for r in st for d in r]
    row_len = [len(r) for st in psteps for r in st]
    plan_sha = hashlib.sha256(json.dumps(psteps).encode()).hexdigest()
    np.savez(TABLES / "labels_main.npz", sem=lab_m.astype(np.int8), shuf=sh_m.astype(np.int8), seen_by_base=seen)
    np.savez(TABLES / "plan.npz", docs=np.asarray(flat, np.int64), row_len=np.asarray(row_len, np.int64),
             rows_per_step=np.int64(ROWS_PER_STEP), plan_sha=np.asarray(plan_sha))
    # steps / tokens per budget and specialist exposure (training tokens per specialist)
    cum = np.cumsum([sum(int(sp.lengths[d]) - 1 for r in st for d in r) for st in psteps])
    bud = {}
    for name, b in BUDGETS.items():
        n = int(np.searchsorted(cum, b) + 1)
        docs = np.asarray([d for st in psteps[:n] for r in st for d in r])
        bud[name] = {"budget": b, "n_steps": n, "tokens": int(cum[n - 1]),
                     "exposure_A": mass(sp, docs, lab_m), "exposure_B": mass(sp, docs, sh_m)}
    out["main"] = {"base_plan_sha256": psha, "base_seen_docs": int(seen.sum()), "pool_docs": len(pool),
                   "pool_tokens": int((sp.lengths[pool] - 1).sum()), "plan_sha256": plan_sha,
                   "plan_steps": len(psteps), "plan_tokens": int(ptot), "budgets": bud, "shuffle": st_m,
                   "unassigned_unseen_docs": int((~seen & (lab_m < 0)).sum())}
    # CALIB-SEARCH
    cs = Split("calib_search")
    lab_c = hard_labels(SEM_CS, SEM_CS_SHA, cs)
    sh_c, st_c = length_stratified_shuffle(cs, lab_c, SHUF_SEED_CS, SHUF_LEN_BINS)
    np.savez(TABLES / "labels_cs.npz", sem=lab_c.astype(np.int8), shuf=sh_c.astype(np.int8))
    parts = {}
    for name, half in (("SEL", 0), ("EVAL", 1)):
        docs = [i for i in paper_half(cs, half) if lab_c[i] >= 0]
        docs = sorted(docs, key=lambda i: u_hash("orv1-eval", f"{cs.papers[i]}|{i}"))
        r = pack_rows(cs.lengths, docs)
        parts[name] = {"docs": np.asarray([d for x in r for d in x], np.int64),
                       "row_len": np.asarray([len(x) for x in r], np.int64)}
        out[name] = {"n_docs": len(docs), "n_rows": len(r), "tokens": int((cs.lengths[docs] - 1).sum()),
                     "n_papers": len({cs.papers[i] for i in docs}), "mass_sem": mass(cs, np.asarray(docs), lab_c),
                     "mass_shuf": mass(cs, np.asarray(docs), sh_c)}
    ps = {n: {cs.papers[i] for i in parts[n]["docs"]} for n in parts}
    if ps["SEL"] & ps["EVAL"]:
        raise guard.GuardError("SEL/EVAL not paper-disjoint")
    np.savez(TABLES / "parts.npz", **{f"{n}_{k}": v for n in parts for k, v in parts[n].items()})
    out["cs_shuffle"] = st_c
    # token features: unigram frequency (main split) and target-token class from the tokenizer vocabulary
    cnt = np.bincount(np.asarray(sp.tokens), minlength=50304)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(TOKENIZER))
    cls = np.zeros(50304, np.int8)
    for i in range(50304):
        s = tk.decode([i], skip_special_tokens=False) if i < tk.get_vocab_size() else ""
        if s.strip() == "":
            cls[i] = 3  # whitespace / newline / empty
        elif any(c.isdigit() for c in s):
            cls[i] = 2  # contains a digit
        elif s[0] == " " and s[1:2].isalpha():
            cls[i] = 0  # word start
        elif s.isalpha():
            cls[i] = 1  # word continuation
        else:
            cls[i] = 4  # punctuation / symbols / other
    np.savez(TABLES / "token_feats.npz", freq=cnt.astype(np.int64), cls=cls)
    out["token_class_counts"] = np.bincount(cls, minlength=5).tolist()
    for f in sorted(TABLES.glob("*.npz")):
        f.chmod(0o400)
        out.setdefault("sha256", {})[f.name] = guard.sha256_file(f)
    guard.atomic_write_json(TABLES / "tables.json", out, mode=0o400)
    print(json.dumps({k: v for k, v in out.items() if k != "sha256"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
