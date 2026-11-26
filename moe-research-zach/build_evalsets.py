#!/usr/bin/env python
"""Held-out evaluation subsets, built ONLY from validation documents (never test, never train):

  topic_subsets.json : single-topic vs multi-topic val documents, by dispersion of frozen sentence embeddings
                       (1 - mean cosine of sentences to the document centroid); bottom / top quartile among val
                       documents with >= 8 sentences. Label-free; used only for post-training analysis.
  transitions        : controlled topic transitions: first 256 tokens of val doc X (no EOS) + first 255 tokens of a
                       semantically distant val doc Y + EOS. Junction position recorded. Segmentations: fixed,
                       online causal semantic (same chunker, same tau), random (same fitted distribution), plus
                       prefix embeddings. Detection delay of the semantic chunker at the known junction is recorded.
CPU only. Requires --execute and MOE_RESEARCH_APPROVE=cpu-model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--n-transitions", type=int, default=150)
    ap.add_argument("--part-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    sd = guard.inside_root(a.segments)
    print(f"eval sets from val docs using {sd}")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    guard.register_self("build_evalsets (CPU)")
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from csr.artifacts import FrozenEncoder, load_lm_tokenizer
    from csr.data import EOS_ID, TokenSplit
    from csr.segcache import add_prefix_embeddings, build_table, embed_doc_sentences, make_detector
    from csr.segment import LengthDistribution, SemanticParams, fixed_starts, random_starts, semantic_starts

    sman = json.loads((sd / "manifest.json").read_text())
    spec = sman["spec"]
    data_dir = guard.ROOT / "artifacts" / "data" / "fwe-v1"
    va = TokenSplit.load(data_dir, "val")
    out = guard.ROOT / "artifacts" / "evalsets" / sman["cache_id"]
    if out.exists():
        print("exists", out)
        return
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.mkdir(parents=True)
    z = np.load(sd / "val_sentences.npz")
    ends, off, embs = z["ends"], z["off"], z["embs"].astype(np.float32)
    disp, cent = {}, {}
    for di in range(va.n_docs):
        e = embs[off[di]:off[di + 1]]
        if len(e) == 0:
            continue
        c = e.mean(0)
        c /= max(np.linalg.norm(c), 1e-12)
        cent[di] = c
        if len(e) >= 8:
            disp[di] = float(1 - (e @ c).mean())
    vals = np.array(list(disp.values()))
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    subsets = {"single_topic": sorted(d for d, v in disp.items() if v <= q1),
               "multi_topic": sorted(d for d, v in disp.items() if v >= q3),
               "rule": "dispersion=1-mean cos(sentence, doc centroid), docs with >=8 sentences; bottom/top quartile",
               "thresholds": [float(q1), float(q3)], "n_eligible": len(disp)}
    (tmp / "topic_subsets.json").write_text(json.dumps(subsets))
    # transitions
    P = a.part_len
    long_docs = [d for d in cent if len(va.doc(d)) > P + 8]
    rng = np.random.default_rng(a.seed)
    xs = rng.choice(long_docs, size=min(a.n_transitions, len(long_docs)), replace=False)
    C = np.stack([cent[d] for d in long_docs])
    docs, meta = [], []
    for x in xs:
        sims = C @ cent[x]
        order = np.argsort(sims)
        y = next(long_docs[i] for i in order if long_docs[i] != x)
        ids = np.concatenate([va.doc(x)[:P], va.doc(y)[:P - 1], [EOS_ID]]).astype(np.int64)
        docs.append(ids)
        meta.append({"x": va.doc_ids[x], "y": va.doc_ids[y], "junction": P, "sim": float(sims[order[0]])})
    lens = np.array([len(d) for d in docs])
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.save(tmp / "transitions_tokens.npy", np.concatenate(docs).astype(np.uint16))
    np.save(tmp / "transitions_offsets.npy", offs)
    (tmp / "transitions_docs.json").write_text(json.dumps({"ids": [f"{m['x']}|{m['y']}" for m in meta], "meta": meta}))
    tr = TokenSplit.load(tmp, "transitions")
    tok = load_lm_tokenizer()
    det = make_detector(tok)
    enc = FrozenEncoder("cpu")
    p = SemanticParams(sman["tau"], spec["min_len"], spec["max_len"])
    sem_ev = [semantic_starts(len(ids), e, x, p, return_events=True)
              for ids, (e, x) in zip(docs, embed_doc_sentences(tok, det, enc, docs))]
    sem = [st for st, _ in sem_ev]
    dist = LengthDistribution(**json.loads((sd / "random_length_distribution.json").read_text()))
    tabs = {"fixed": build_table("fixed", tr, [fixed_starts(int(L), spec["fixed_size"]) for L in lens]),
            "semantic": build_table("semantic", tr, sem),
            "random": build_table("random", tr, [random_starts(int(L), k, dist, spec["random_seed"])
                                                 for L, k in zip(lens, tr.doc_ids)])}
    for name, t in tabs.items():
        if name != "random":
            add_prefix_embeddings(t, tr, tok, enc, spec["warmup"])
        t.save(tmp / f"transitions_{name}.npz")
    delays = []
    for _, ev in sem_ev:
        after = [e_[1] for e_ in ev if e_[0] == "semantic" and e_[1] >= P]
        delays.append(after[0] - P if after else None)
    found = [d for d in delays if d is not None]
    man = {"cache_id": sman["cache_id"], "topic_subsets": {k: len(v) for k, v in subsets.items() if isinstance(v, list)},
           "n_transitions": len(docs), "semantic_boundary_after_junction_fraction": len(found) / len(docs),
           "semantic_detection_delay_tokens": {"mean": float(np.mean(found)) if found else None,
                                               "median": float(np.median(found)) if found else None},
           "created": guard.now()}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out)
    guard.ledger("EVALSETS_BUILD", json.dumps(man))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
