"""Dedup detects planted exact and near duplicates; splits are document-level and deterministic."""
from csr.data import dedup, split_of


def test_dedup_planted():
    base = " ".join(f"word{i}" for i in range(400))
    near = base.replace("word17 ", "wordX ", 1)
    other = " ".join(f"tok{i}" for i in range(400))
    ids = ["a", "b", "c", "d"]
    keep, rep = dedup(ids, [base, base.upper(), near, other])
    assert keep.tolist() == [True, False, False, True]
    assert rep["exact_dups"] == 1 and rep["near_dups"] == 1


def test_split_deterministic_and_document_level():
    s = [split_of(f"<urn:uuid:{i}>") for i in range(4000)]
    assert s == [split_of(f"<urn:uuid:{i}>") for i in range(4000)]
    frac = {k: s.count(k) / len(s) for k in set(s)}
    assert abs(frac["train"] - 0.80) < 0.03 and abs(frac["test"] - 0.075) < 0.02
