import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)


def pytest_configure(config):
    # enforced for every pytest session; not triggered when gpu_checks.py imports the helpers below
    assert not torch.cuda.is_available(), "tests must run with CUDA hidden"

from csr.model import CSRModel, ModelConfig, RoutingConfig  # noqa: E402
from csr.segment import SentenceDetector  # noqa: E402


def tiny_cfg(**kw):
    base = dict(vocab_size=97, n_layer=3, d_model=32, n_head=4, dense_ffn=48, moe_layers=(1, 2), n_experts=8,
                top_k=2, expert_ffn=16, shared_ffn=24, max_seq_len=128)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return CSRModel(tiny_cfg()).double().eval()


def single_doc_batch(ids, starts):
    from csr.segment import starts_to_arrays
    L = len(ids)
    cidx, cpos = starts_to_arrays(L, starts)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64)).unsqueeze(0)
    tgt = np.full(L, -100)
    tgt[:-1] = ids[1:]
    return {"input_ids": t(ids), "doc_ids": t(np.zeros(L)), "positions": t(np.arange(L)), "targets": t(tgt),
            "chunk_ids": t(cidx), "chunk_pos": t(cpos), "gdoc": t(np.zeros(L)), "chunk_key": t(cidx)}


# ----- fake tokenizer / encoder for segmentation tests: tokens 0..9 are "words" of topic A, 10..19 topic B,
# token 50 ends a sentence.
END = 50


def fake_detector(vocab=97):
    return SentenceDetector(lambda i: "." if i == END else "w", vocab, min_sentence_tokens=2)


def fake_embed(list_of_ids):
    out = []
    for ids in list_of_ids:
        v = np.zeros(4)
        for t in ids:
            if t < 10:
                v[0] += 1
            elif t < 20:
                v[1] += 1
            elif t < 30:
                v[2] += 1
            else:
                v[3] += 0.1
        out.append(v / max(np.linalg.norm(v), 1e-12))
    return np.asarray(out)


def topic_doc(rng, n_sent_per_topic=(12, 12, 12), sent_len=(5, 9)):
    ids = []
    for ti, n in enumerate(n_sent_per_topic):
        for _ in range(n):
            L = rng.integers(*sent_len)
            ids.extend((rng.integers(0, 10, size=L) + 10 * (ti % 3)).tolist())
            ids.append(END)
    return np.asarray(ids, dtype=np.int64)
