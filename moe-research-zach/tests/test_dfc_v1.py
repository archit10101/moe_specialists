"""csr-domain-forecast-curriculum-v1 CPU correctness tests (CUDA hidden). Implementation checks, not a gate."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from csr.dfc_lib import (CACHE, DATA, FROZEN, MAIN_TOKENS, N_DOM, PHASE_TOKENS, SplitLocked, interleave,  # noqa: E402
                         load_cache, load_split, mi_from_counts)
from csr.dfc_model import (CONDITIONS, DERANGE, DFCModel, DraftModel, draft_forecasts, provisional_group,  # noqa: E402
                           to_plain_csr)
from csr.model import CSRModel, ModelConfig, RoutingConfig  # noqa: E402

HAVE_DATA = (DATA / "manifest.json").exists()
HAVE_CACHE = (CACHE / "main_correct.npy").exists() and (CACHE / "calib_correct.npy").exists()
QUOTAS = {"val": 1_000_000, "calib": 1_500_000, "draft": 6_000_000, "main_p1": 9_000_000, "main_p2": 2_250_000,
          "main_p3": 11_250_000}

TINY = dict(vocab_size=64, n_layer=3, d_model=32, n_head=4, dense_ffn=64, moe_layers=(1, 2), n_experts=32, top_k=4,
            expert_ffn=8, shared_ffn=16, max_seq_len=64, tie_embeddings=True)
TINY_DRAFT = dict(vocab_size=64, n_layer=2, d_model=16, n_head=2, dense_ffn=32, moe_layers=(), n_experts=1, top_k=1,
                  expert_ffn=1, shared_ffn=0, max_seq_len=64, tie_embeddings=True)


def tiny_main(cond, seed=0):
    torch.manual_seed(seed)
    return DFCModel(ModelConfig(**TINY), cond, seed).double()


def tiny_draft(seed=0, nonzero_adapters=True):
    torch.manual_seed(seed)
    m = DraftModel(ModelConfig(**TINY_DRAFT)).double()
    if nonzero_adapters:
        with torch.no_grad():
            for la in m.adapters:
                for ad in la:
                    ad.up.weight.normal_(0, 0.5)
    return m.eval()


def one_doc_batch(ids, dom):
    T = len(ids)
    return {"input_ids": torch.tensor([ids]), "doc_ids": torch.zeros(1, T, dtype=torch.long),
            "positions": torch.arange(T).unsqueeze(0), "domain": torch.full((1, T), dom)}


def sha_params(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters(), key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().double().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ data invariants (real data)
@pytest.mark.skipif(not HAVE_DATA, reason="data not built")
def test_split_isolation_quotas_and_dedup():
    man = json.loads((DATA / "manifest.json").read_text())
    sets = {}
    for sp in ("calib", "draft", "main", "val"):  # val: only the hash list is read, never tokens
        dj = json.loads((DATA / f"{sp}_docs.json").read_text())
        assert len(dj["ids"]) == len(set(dj["ids"])), f"duplicate segment id inside {sp}"
        assert all(i.split("#")[0] == p for i, p in zip(dj["ids"], dj["papers"]))
        sets[sp] = set(dj["papers"])  # isolation is by PAPER hash
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (sets[names[i]] & sets[names[j]]), (names[i], names[j])
    for dom, meta in man["per_domain"].items():
        for b, q in QUOTAS.items():
            assert meta[b]["valid_tokens"] == q, (dom, b)
    for sp, q in (("calib", 7_500_000), ("draft", 30_000_000), ("main", MAIN_TOKENS), ("val", 5_000_000)):
        assert man["splits"][sp]["valid_tokens"] == q
    inv = json.loads((ROOT / "artifacts/domain-forecast-curriculum-v1/inventory.json").read_text())
    assert inv["selected"] == man["domains"] and len(man["domains"]) == 5
    for d in man["domains"]:
        assert inv["candidates"][d]["eligible"]
    for d in inv["candidates"]:
        assert inv["candidates"][d]["verified"]
        assert inv["candidates"][d]["members"]["test"]["decoded"] is False


def test_exact_and_near_dedup_mechanics():
    import dfc_data as D
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(5000)]
    base = [" ".join(rng.choice(words, 900)) for _ in range(30)]
    near = base[3].split(" ")
    near[10] = "zzz"
    near[500] = "yyy"
    texts = base + [" ".join(near), base[7].upper()]  # near dup of 3; exact (normalised) dup of 7
    hs = [hashlib.sha256(D.norm(t).encode()).hexdigest() for t in texts]
    assert hs[-1] == hs[7]
    sigs = np.stack([D.minhash(t) for t in texts])
    uf, _ = D.near_dedup(sigs)
    assert uf.find(30) == uf.find(3)
    roots = {uf.find(i) for i in range(30)}
    assert len(roots) == 30  # unrelated docs never merged


@pytest.mark.skipif(not HAVE_DATA, reason="data not built")
def test_main_plan_balance_phases_and_schedule():
    from csr.dfc_lib import main_plan
    p = main_plan(0)
    pdp = p.per_domain_phase_tokens()
    for d in range(N_DOM):
        for ph, tot in PHASE_TOKENS.items():
            assert pdp[f"p{ph}_d{d}"] == tot // N_DOM
    ph = p.row_phase
    assert (np.diff(ph) >= 0).all()  # P1 rows, then P2, then P3
    s = p.row_strength
    assert (s[ph == 1] == 1).all() and (s[ph == 3] == 0).all()
    s2 = s[ph == 2]
    assert (s2 > 0).all() and (s2 < 1).all() and (np.diff(s2) < 0).all()
    cum = np.cumsum(p.row_valid)
    assert (s[cum > 56_250_000 + 1024] == 0).all()  # zero influence after 50%
    # interleaving: every 32-row step mixes domains (no sequential domain training)
    rd = np.asarray([p.doms[r[0]] for r in p.rows])
    for st in p.steps[:-1]:
        c = np.bincount(rd[st], minlength=N_DOM)
        assert c.min() >= 4 and c.max() <= 9, c
    # deterministic + identical across conditions within seed
    if HAVE_CACHE:
        q = main_plan(0, "correct")
        assert q.fingerprint() == p.fingerprint()
    assert main_plan(1).fingerprint() != p.fingerprint()


@pytest.mark.skipif(not HAVE_DATA, reason="data not built")
def test_alldata_and_compute_plans_exact():
    from csr.dfc_lib import alldata_plan, compute_plan
    a = alldata_plan(0)
    L = a.split.lengths()
    per = np.zeros(N_DOM, dtype=np.int64)
    for r in a.rows:
        for di in r:
            per[a.doms[di]] += L[di] - 1
    assert (per == 28_500_000).all()
    c = compute_plan(0, 125_000_000)
    L = c.split.lengths()
    per = np.zeros(N_DOM, dtype=np.int64)
    for r in c.rows:
        for di in r:
            per[c.doms[di]] += L[di] - 1
    assert (per == 25_000_000).all() and c.cycles == 1


def test_interleave_fractional():
    rows, doms = interleave([[["a"]] * 10, [["b"]] * 5])
    assert len(rows) == 15 and doms[:3] == [0, 1, 0]


# ------------------------------------------------------------------ split locks
def test_split_locks():
    with pytest.raises(SplitLocked):
        load_split("test")
    if not FROZEN.exists():
        with pytest.raises(SplitLocked):
            load_split("val")
    with pytest.raises(SplitLocked):
        load_cache("val", "correct")


# ------------------------------------------------------------------ draft adapters and causal forecasts
def test_draft_adapter_selection():
    m = tiny_draft()
    ids = list(np.random.default_rng(0).integers(2, 64, 20))
    b = one_doc_batch(ids, 2)
    y0 = m(b).logits
    with torch.no_grad():
        for d in (0, 1, 3, 4):
            for li in range(len(m.adapters)):
                m.adapters[li][d].up.weight.add_(1.0)
    assert torch.equal(m(b).logits, y0)  # other domains' adapters never used
    with torch.no_grad():
        m.adapters[0][2].up.weight.add_(1.0)
    assert not torch.allclose(m(b).logits, y0)  # own adapter used
    z = DraftModel(ModelConfig(**TINY_DRAFT)).double()
    plain = CSRModel(ModelConfig(**TINY_DRAFT)).double()
    plain.load_state_dict({k: v for k, v in z.state_dict().items() if not k.startswith("adapters")})
    assert torch.allclose(z(b).logits, plain(b, RoutingConfig()).logits)  # zero-init up: identity at init


def _greedy_from_prefix(m, ids, dom, g=4):
    seq = list(ids)
    out = []
    for _ in range(g):
        nxt = int(m(one_doc_batch(seq, dom)).logits[0, -1].argmax())
        out.append(nxt)
        seq.append(nxt)
    return out


def test_forecasts_equal_autoregressive_and_causal():
    m = tiny_draft()
    ids = list(np.random.default_rng(1).integers(2, 64, 24))
    b = one_doc_batch(ids, 1)
    fc = draft_forecasts(m, b["input_ids"], b["doc_ids"], b["positions"], b["domain"])
    for t in (0, 5, 17, 23):
        assert fc[0, t].tolist() == _greedy_from_prefix(m, ids[:t + 1], 1)
    # perturb the future: forecasts at positions <= t unchanged
    for t in (3, 11, 20):
        ids2 = ids[:t + 1] + list(np.random.default_rng(t).integers(2, 64, len(ids) - t - 1))
        b2 = one_doc_batch(ids2, 1)
        fc2 = draft_forecasts(m, b2["input_ids"], b2["doc_ids"], b2["positions"], b2["domain"])
        assert torch.equal(fc[0, :t + 1], fc2[0, :t + 1])


def test_cache_invariant_to_batching():
    import dfc_cache
    from csr.dfc_lib import MemSplit
    m = tiny_draft()
    rng = np.random.default_rng(2)
    docs = [rng.integers(2, 64, rng.integers(5, 40)).astype(np.uint16) for _ in range(11)]
    off = np.concatenate([[0], np.cumsum([len(d) for d in docs])])
    sp = MemSplit(np.concatenate(docs), off, [str(i) for i in range(11)])
    doms = rng.integers(0, 5, 11)
    dev = torch.device("cpu")
    m = m.float()
    a = dict(dfc_cache.canonical_forecasts(m, sp, doms, list(range(11)), lambda d: d, dev, batch=4, T=64))
    b = dict(dfc_cache.canonical_forecasts(m, sp, doms, list(range(11))[::-1], lambda d: d, dev, batch=3, T=64))
    for k in a:
        assert np.array_equal(a[k], b[k])
    c = dict(dfc_cache.canonical_forecasts(m, sp, doms, list(range(11)), lambda d: DERANGE[d], dev, batch=4, T=64))
    assert any(not np.array_equal(a[k], c[k]) for k in a)


def test_derangement():
    assert all(DERANGE[d] != d for d in range(5)) and sorted(DERANGE.values()) == list(range(5))


# ------------------------------------------------------------------ main model
def _batch(B=2, T=24, seed=0, fc=True):
    rng = np.random.default_rng(seed)
    b = {"input_ids": torch.tensor(rng.integers(2, 64, (B, T))), "doc_ids": torch.zeros(B, T, dtype=torch.long),
         "positions": torch.arange(T).repeat(B, 1), "domain": torch.tensor(rng.integers(0, 5, (B, 1))).repeat(1, T)}
    if fc:
        b["fc"] = torch.tensor(rng.integers(2, 64, (B, T, 4)))
    return b


def test_identical_init_across_conditions_and_backbone():
    shas = {c: sha_params(tiny_main(c, seed=3)) for c in CONDITIONS}
    assert len(set(shas.values())) == 1
    torch.manual_seed(3)
    plain = CSRModel(ModelConfig(**TINY)).double()
    m = tiny_main("SOFT", seed=3)
    for k, v in plain.state_dict().items():
        assert torch.equal(v, m.state_dict()[k]), k
    assert sha_params(tiny_main("SOFT", seed=4)) != shas["SOFT"]


def test_strength_zero_is_exact_token4_and_needs_no_draft_or_labels():
    for c in CONDITIONS:
        m = tiny_main(c)
        plain = to_plain_csr(m).double()
        b = _batch()
        ref = plain({k: b[k] for k in ("input_ids", "doc_ids", "positions")}, RoutingConfig())
        nolab = {k: b[k] for k in ("input_ids", "doc_ids", "positions")}
        for s in (None, 0.0, torch.zeros(2)):
            out = m(nolab, strength=s)
            assert torch.equal(out.logits, ref.logits)
            for r1, r2 in zip(out.routing, ref.routing):
                assert torch.equal(r1["topi"], r2["topi"])


def test_curriculum_changes_routing_only_when_active_and_rows_independent():
    m = tiny_main("SOFT")
    b = _batch()
    base = m(b, strength=0.0).logits
    act = m(b, strength=1.0).logits
    assert not torch.allclose(base, act)
    mixed = m(b, strength=torch.tensor([1.0, 0.0])).logits
    assert torch.equal(mixed[1], base[1]) and torch.allclose(mixed[0], act[0])
    tok = tiny_main("TOKEN-MAIN")
    assert torch.equal(tok(b, strength=1.0).logits, tok(b, strength=0.0).logits)  # TOKEN conditions ignore strength


def test_four_distinct_experts_all_eligible_and_shared_identical():
    for c in ("SOFT", "LABEL", "FORECAST", "WRONG"):
        m = tiny_main(c)
        b = _batch(B=4, T=32)
        out = m(b, strength=1.0)
        for r in out.routing:
            ti = r["topi"]
            assert ti.shape[-1] == 4
            assert (ti.sort(-1).values.diff(dim=-1) > 0).all()  # 4 distinct
        # eligibility: a general (non-provisional) expert with a dominant token logit is still selected under the prior
        with torch.no_grad():
            for blk in m.moe_blocks():
                blk.ffn.router.weight[27] = 0
                blk.ffn.router.weight[27, 0] = 50.0
        b2 = _batch(B=2, T=16)
        out = m(b2, strength=1.0)
        hit = torch.stack([(r["topi"] == 27).any(-1) for r in out.routing]).float().mean()
        assert hit > 0.4
    a, s = tiny_main("TOKEN-MAIN"), tiny_main("SOFT")
    x = torch.randn(10, 32, dtype=torch.float64)
    assert torch.equal(a.blocks[1].ffn.shared(x), s.blocks[1].ffn.shared(x))


def test_prior_favours_provisional_group_at_init():
    m = tiny_main("LABEL")
    b = _batch(B=4, T=32)
    out = m(b, strength=1.0)
    d = b["domain"][:, 0]
    frac = []
    for r in out.routing:
        for i in range(4):
            g = set(provisional_group(int(d[i])))
            frac.append(np.mean([len(set(x) & g) / 4 for x in r["topi"][i].tolist()]))
    assert np.mean(frac) > 4 / 32


def test_forecast_stopgrad_and_grads_only_when_active():
    m = tiny_main("SOFT")
    b = _batch()
    m.forecast_repr(b["fc"]).sum().backward()
    assert m.tok_emb.weight.grad is None or float(m.tok_emb.weight.grad.abs().sum()) == 0.0
    m.zero_grad(set_to_none=True)
    m(b, strength=0.0).logits.sum().backward()
    assert all(p.grad is None for n, p in m.named_parameters() if n.startswith(("fc_", "rt_")))
    m.zero_grad(set_to_none=True)
    m(b, strength=1.0).logits.sum().backward()
    assert m.rt_prior.grad is not None and m.fc_gru.weight_ih_l0.grad is not None
    lab = tiny_main("LABEL")
    lab(b, strength=1.0).logits.sum().backward()
    assert lab.fc_gru.weight_ih_l0.grad is None and lab.rt_wf.grad is None  # LABEL: no forecast path
    fo = tiny_main("FORECAST")
    fo(b, strength=1.0).logits.sum().backward()
    assert fo.rt_prior.grad is None  # FORECAST: no prior


def test_main_model_future_perturbation_with_forecasts():
    d = tiny_draft()
    m = tiny_main("SOFT")
    ids = list(np.random.default_rng(5).integers(2, 64, 28))
    for t in (4, 13, 22):
        outs = []
        for variant in range(2):
            x = ids if variant == 0 else ids[:t + 1] + list(np.random.default_rng(t).integers(2, 64, 27 - t))
            b = one_doc_batch(x, 3)
            b["fc"] = draft_forecasts(d, b["input_ids"], b["doc_ids"], b["positions"], b["domain"])
            outs.append(m(b, strength=1.0))
        assert torch.equal(outs[0].logits[0, :t + 1], outs[1].logits[0, :t + 1])
        for r1, r2 in zip(outs[0].routing, outs[1].routing):
            assert torch.equal(r1["topi"][0, :t + 1], r2["topi"][0, :t + 1])


def test_mi_metric():
    C = np.zeros((5, 32))
    for d in range(5):
        C[d, provisional_group(d)] = 100
    mi, hd, hde = mi_from_counts(C)
    assert abs(mi - hd) < 1e-9 and abs(hd - np.log(5)) < 1e-9
    mi0, _, _ = mi_from_counts(np.ones((5, 32)))
    assert abs(mi0) < 1e-12


# ------------------------------------------------------------------ real-script resume equality (CPU, SOFT config)
@pytest.mark.skipif(not (HAVE_DATA and HAVE_CACHE and (ROOT / "configs/domain-forecast-curriculum-v1/main/SOFT_s0.json")
                         .exists()), reason="needs data, caches and configs")
def test_interrupted_resumed_equals_uninterrupted(tmp_path_factory):
    base = ROOT / ".tmp/domain-forecast-curriculum-v1/cpu_test_runs" / f"t{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    cfg = "configs/domain-forecast-curriculum-v1/main/SOFT_s0.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MOE_RESEARCH_APPROVE="cpu-model")

    def run(slot, extra):
        p = subprocess.run([sys.executable, "dfc_train.py", "--config", cfg, "--device", "cpu", "--execute",
                            "--runs-dir", str(base), "--slot-dir", str(base / slot), "--micro-batch-rows", "8"] + extra,
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=3000)
        rd = [ln.split(" ", 1)[1] for ln in p.stdout.splitlines() if ln.startswith("RUN_DIR ")]
        return p.returncode, (rd[-1] if rd else None), p.stdout[-3000:] + p.stderr[-3000:]
    rc, ra, log = run("A", ["--max-steps", "3"])
    assert rc == 75, log
    rc, rb, log = run("B", ["--max-steps", "1"])
    assert rc == 75, log
    rc, rb2, log = run("B", ["--max-steps", "3", "--run-dir", rb])
    assert rc == 75 and rb2 == rb, log
    sa = torch.load(base / "A" / "resume_slot.pt", map_location="cpu", weights_only=False)
    sb = torch.load(base / "B" / "resume_slot.pt", map_location="cpu", weights_only=False)
    assert sa["st"]["next_step"] == sb["st"]["next_step"] == 3 and sa["st"]["seen"] == sb["st"]["seen"]
    for k in sa["model"]:
        assert torch.equal(sa["model"][k], sb["model"][k]), k
