"""Condition F v2 (random hazard closer): determinism, content-independence, causality, identical eligibility
rules with semantic D, document boundaries."""
import inspect

import numpy as np
import pytest
import torch

from conftest import END, fake_detector, fake_embed, topic_doc
from csr.segment import (HazardPolicy, SemanticParams, chunk_lengths, hazard_starts, semantic_starts,
                         sentence_spans, starts_to_arrays)

MIN, MAX = 24, 64
EDGES = [0.0, 150.0, float("inf")]


def policy(val=0.3, n_strata=2):
    nb = (MAX - MIN) // 8
    rng = np.random.default_rng(0)
    return HazardPolicy(EDGES, 8, MIN, MAX, (rng.uniform(0.05, val * 2, size=(n_strata, nb))).tolist())


def docs(n=8):
    return [topic_doc(np.random.default_rng(s), n_sent_per_topic=(8, 14, 6)) for s in range(n)]


def sem(det, ids, tau):
    ends = det.sentence_ends(ids)
    embs = fake_embed([ids[s:e + 1] for s, e in sentence_spans(ends)])
    return semantic_starts(len(ids), ends, embs, SemanticParams(tau, MIN, MAX), return_events=True)


def test_deterministic_from_seed():
    det, pol = fake_detector(), policy()
    for ids in docs():
        e = det.sentence_ends(ids)
        a = hazard_starts(len(ids), e, pol, "k", 7)
        assert a == hazard_starts(len(ids), e, pol, "k", 7)
    outs7 = [hazard_starts(len(i), det.sentence_ends(i), pol, "k", 7) for i in docs()]
    outs8 = [hazard_starts(len(i), det.sentence_ends(i), pol, "k", 8) for i in docs()]
    assert outs7 != outs8


def test_reads_no_semantic_features():
    # the interface admits no text / embeddings / similarities
    params = set(inspect.signature(hazard_starts).parameters)
    assert params == {"L", "ends", "policy", "doc_key", "seed", "decide", "return_events"}
    # content-independence: completely different tokens with identical length and sentence-end positions
    det, pol = fake_detector(), policy()
    ids = docs(1)[0]
    other = np.where(ids == END, END, (ids + 37) % 49)  # different words, same sentence ends
    assert np.array_equal(det.sentence_ends(ids), det.sentence_ends(other))
    assert hazard_starts(len(ids), det.sentence_ends(ids), pol, "k", 3) == \
        hazard_starts(len(other), det.sentence_ends(other), pol, "k", 3)
    # the fitting/building modules never import the encoder or the semantic loss
    import ast
    import csr.random_hazard as rh
    import csr.segment as sg
    tree = ast.parse(inspect.getsource(rh))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} | \
            {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    banned = {"FrozenEncoder", "embed", "embs", "embed_doc_sentences", "semantic_loss", "relational_kl", "u",
              "tau", "tau_s", "sim", "artifacts", "segcache", "semantic_starts", "SemanticParams"}
    assert not (names & banned), names & banned
    fn = ast.parse(inspect.getsource(sg.hazard_starts))
    fn_names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not (fn_names & (banned | {"rep", "dot", "norm"})), fn_names & banned


def test_future_tokens_do_not_affect_earlier_boundaries():
    det, pol = fake_detector(), policy()
    for k, ids in enumerate(docs()):
        cut = len(ids) // 2
        ids2 = ids.copy()
        ids2[cut + 1:] = np.random.default_rng(50 + k).integers(0, 30, size=len(ids) - cut - 1)
        ids2[cut + 1::7] = END  # different future sentence structure, same length (same stratum)
        s1 = hazard_starts(len(ids), det.sentence_ends(ids), pol, "k", 5)
        s2 = hazard_starts(len(ids2), det.sentence_ends(ids2), pol, "k", 5)
        assert [s for s in s1 if s <= cut + 1] == [s for s in s2 if s <= cut + 1]


@pytest.mark.parametrize("tau,hval", [(2.0, 1.0), (-2.0, 0.0)])
def test_identical_eligibility_extremes(tau, hval):
    """Always-close / never-close: semantic D with tau=+2 / -2 equals random with hazard 1 / 0."""
    det = fake_detector()
    nb = (MAX - MIN) // 8
    pol = HazardPolicy(EDGES, 8, MIN, MAX, [[hval] * nb, [hval] * nb])
    for ids in docs(12):
        s_sem, _ = sem(det, ids, tau)
        assert hazard_starts(len(ids), det.sentence_ends(ids), pol, "k", 1) == s_sem


def test_identical_eligibility_scripted_replay():
    """Feeding D's own closure decisions through the random state machine reproduces D exactly, and the random
    closer is offered a decision at exactly the positions where D evaluated similarity."""
    det = fake_detector()
    pol = policy()
    for ids in docs(12):
        s_sem, ev = sem(det, ids, 0.5)
        sset = set(s_sem[1:])
        asked = []

        def decide(s, cur_len, t_end):
            asked.append(t_end + 1)
            return (t_end + 1) in sset
        assert hazard_starts(len(ids), det.sentence_ends(ids), pol, "k", 0, decide=decide) == s_sem
        sem_closed = {e[1] for e in ev if e[0] == "semantic"}
        assert sem_closed <= set(asked)


def test_document_boundaries_respected():
    det, pol = fake_detector(), policy()
    for ids in docs():
        L = len(ids)
        st = hazard_starts(L, det.sentence_ends(ids), pol, "k", 9)
        assert st[0] == 0 and all(0 < s < L for s in st[1:]) and st == sorted(set(st))
        ls = chunk_lengths(L, st)
        assert ls.sum() == L and ls.max() <= MAX and all(x >= MIN for x in ls[:-1])
    # in packed rows every document starts a new chunk at chunk_pos 0
    from csr.data import TokenSplit, make_batch
    ds = docs(3)
    off = np.concatenate([[0], np.cumsum([len(d) for d in ds])])
    sp = TokenSplit(np.concatenate(ds).astype(np.uint16), off, ["a", "b", "c"])
    starts = {i: hazard_starts(len(d), det.sentence_ends(d), pol, k, 9) for i, (d, k) in enumerate(zip(ds, "abc"))}
    b = make_batch([[0, 1, 2]], sp, int(off[-1]), starts_of=lambda i: starts[i])
    for i in range(3):
        assert b["chunk_pos"][0, off[i]] == 0
        if i:
            assert b["chunk_ids"][0, off[i]] != b["chunk_ids"][0, off[i] - 1]
