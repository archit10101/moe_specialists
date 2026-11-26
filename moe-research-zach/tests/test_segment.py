"""Gates 1, 6, 11: causal segmentation."""
import numpy as np
import pytest

from conftest import END, fake_detector, fake_embed, topic_doc
from csr.segment import (OnlineSemanticChunker, SemanticParams, SentenceDetector, chunk_lengths,
                         fit_length_distribution, fixed_starts, random_starts, semantic_starts, sentence_spans,
                         starts_to_arrays)

P = SemanticParams(tau=0.5, min_len=24, max_len=64)


def batch_semantic(det, ids, p=P, L=None):
    ends = det.sentence_ends(ids)
    embs = fake_embed([ids[s:e + 1] for s, e in sentence_spans(ends)])
    return semantic_starts(len(ids) if L is None else L, ends, embs, p, return_events=True)


def test_sentence_detector_text_rule():
    f = SentenceDetector.text_ends_sentence
    assert f(".") and f("?\"") and f(" end!") and f("\n") and f(".)")
    assert not f("Mr") and not f("word") and not f(",")


def test_sentence_min_tokens():
    det = fake_detector()  # min_sentence_tokens=2
    ids = np.array([1, END, 2, 3, END, 4, 5, 6, END])
    assert det.sentence_ends(ids).tolist() == [1, 4, 8]
    det.min_sentence_tokens = 3  # the 2-token first "sentence" is merged into the next one
    assert det.sentence_ends(ids).tolist() == [4, 8]


def test_fixed_and_arrays():
    assert fixed_starts(300, 128) == [0, 128, 256]
    c, p = starts_to_arrays(10, [0, 4, 7])
    assert c.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2] and p.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]


def test_semantic_finds_topic_shifts_within_length_limits():
    rng = np.random.default_rng(0)
    det = fake_detector()
    for seed in range(5):
        ids = topic_doc(np.random.default_rng(seed))
        starts, ev = batch_semantic(det, ids)
        ls = chunk_lengths(len(ids), starts)
        assert all(P.min_len <= x <= P.max_len for x in ls[:-1]), ls
        assert ls[-1] <= P.max_len
        assert any(e[0] == "semantic" for e in ev)


def test_streaming_equals_batch_and_is_prefix_causal():
    det = fake_detector()
    for seed in range(6):
        ids = topic_doc(np.random.default_rng(seed), n_sent_per_topic=(8, 15, 6))
        starts_b, _ = batch_semantic(det, ids)
        ch = OnlineSemanticChunker(det, fake_embed, P, L=len(ids))
        flags = [ch.push(int(t)) for t in ids]
        assert ch.starts == starts_b
        # decision "token t+1 starts a chunk" is reported when t is pushed
        assert [i + 1 for i, f in enumerate(flags) if f] == starts_b[1:]
        # prefix causality: changing tokens after `cut` leaves boundaries <= cut+1 unchanged
        cut = len(ids) // 2
        ids2 = ids.copy()
        ids2[cut + 1:] = np.random.default_rng(100 + seed).integers(20, 30, size=len(ids) - cut - 1)
        ch2 = OnlineSemanticChunker(det, fake_embed, P, L=None)
        for t in ids2[:cut + 1]:
            ch2.push(int(t))
        ch1 = OnlineSemanticChunker(det, fake_embed, P, L=None)
        for t in ids[:cut + 1]:
            ch1.push(int(t))
        assert ch1.starts == ch2.starts
        assert [s for s in starts_b if s <= cut] == [s for s in ch1.starts if s <= cut]


def test_embed_fn_never_sees_unobserved_tokens():
    det = fake_detector()
    ids = topic_doc(np.random.default_rng(3))
    seen = []

    def spy(lst):
        seen.append(list(lst[0]))
        return fake_embed(lst)

    ch = OnlineSemanticChunker(det, spy, P, L=len(ids))
    for t, tok in enumerate(ids):
        n_before = len(seen)
        ch.push(int(tok))
        for s in seen[n_before:]:
            assert s == ids[t - len(s) + 1:t + 1].tolist()  # embedded text ends at the current token


def test_no_boundary_after_last_token_and_forced_max():
    det = fake_detector()
    ids = np.arange(200) % 9 + 1  # no sentence ends at all -> forced boundaries only
    starts, ev = batch_semantic(det, ids)
    assert starts == [0, 64, 128, 192]
    assert all(e[0] == "forced" for e in ev)


def test_random_boundaries_deterministic_and_content_free():
    rng = np.random.default_rng(0)
    doc_lengths = rng.integers(100, 1000, size=300)
    sem = [fixed_starts(int(L), int(rng.integers(64, 200))) for L in doc_lengths]
    dist = fit_length_distribution(doc_lengths, sem, n_bins=4, min_len=64, max_len=256)
    a = random_starts(700, "doc-xyz", dist, seed=7)
    assert a == random_starts(700, "doc-xyz", dist, seed=7)
    assert a != random_starts(700, "doc-xyz", dist, seed=8)
    ls = chunk_lengths(700, a)
    support = set(x for b in dist.lengths for x in b)
    assert all(x in support for x in ls[:-1])
    # the random rule never receives the token ids: same (L, key, seed) -> same boundaries by construction


def test_fit_excludes_final_chunks():
    dist = fit_length_distribution([100, 100], [[0, 70], [0, 80]], n_bins=1, min_len=64, max_len=256)
    assert sorted(dist.lengths[0]) == [70, 80]
