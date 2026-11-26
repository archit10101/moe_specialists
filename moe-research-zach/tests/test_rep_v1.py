"""csr-representative-v1: tied embeddings, parameter counts, A-F matched initialization, F-v3 random control
(content-free, D's state machine, acceptance), dedup against a reference, batched sentence embedding, tau / lambda
selection rules, 100M-token accounting, checkpoint round trip and evaluation pairing. CPU only, CUDA hidden."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conftest import single_doc_batch, tiny_cfg  # noqa: E402
from csr.model import CSRModel, ModelConfig, RoutingConfig, count_parameters  # noqa: E402

REP_DATA = ROOT / "artifacts" / "data" / "fwe-rep-v1"
REP_CFG = sorted((ROOT / "configs" / "rep-v1").glob("seed0/base.json")) or sorted((ROOT / "configs" / "rep-v1").glob("precal/base.json"))


def rep_model_cfg():
    from rep_make_configs import model_block
    return ModelConfig.from_dict(model_block())


# ------------------------------------------------------------------ tied embeddings
def tied_tiny(**kw):
    return tiny_cfg(tie_embeddings=True, **kw)


def test_tied_shared_storage_and_unique_parameters():
    torch.manual_seed(0)
    m = CSRModel(tied_tiny())
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    names = [n for n, _ in m.named_parameters()]
    assert "tok_emb.weight" in names and "lm_head.weight" not in names
    with torch.no_grad():
        m.tok_emb.weight[3, 0] = 123.0
    assert float(m.lm_head.weight[3, 0]) == 123.0
    untied = CSRModel(tiny_cfg())
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert count_parameters(untied)["total"] - count_parameters(m)["total"] == 97 * 32


def test_tied_no_duplicate_optimizer_parameters():
    from csr.train_lib import make_optimizer
    m = CSRModel(tied_tiny())
    cfg = {"train": {"weight_decay": 0.1, "lr": 1e-3, "betas": [0.9, 0.95]}}
    opt = make_optimizer(m, cfg, torch.device("cpu"))
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == count_parameters(m)["total"]


def test_tied_gradient_flows_from_both_uses():
    torch.manual_seed(0)
    m = CSRModel(tied_tiny())
    ids = (np.arange(40) * 2) % 90 + 3  # odd ids 3..81
    b = single_doc_batch(ids, [0, 20])
    out = m(b, RoutingConfig("token", 4, 4))
    (out.logits[..., :50].pow(2).sum()).backward()
    g = m.tok_emb.weight.grad
    absent = sorted(set(range(50)) - set(ids.tolist()))
    assert absent and g[absent].abs().sum() > 0  # rows unseen as inputs get gradient only through the LM head
    assert g[sorted(set(ids.tolist()) & set(range(50, 97)))].abs().sum() > 0  # input-only rows: embedding path


def test_tied_state_dict_roundtrip_identical_logits():
    torch.manual_seed(0)
    m = CSRModel(tied_tiny()).double().eval()
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "config": {"model": m.cfg.to_dict()}}, buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=True)
    assert sd["config"]["model"]["tie_embeddings"] is True
    m2 = CSRModel(ModelConfig.from_dict(sd["config"]["model"])).double().eval()
    m2.load_state_dict(sd["model"], strict=True)
    assert m2.lm_head.weight is m2.tok_emb.weight
    ids = (np.arange(60) * 7) % 90 + 3
    b = single_doc_batch(ids, [0, 18, 40])
    for mode in ("token", "pool"):
        rc = RoutingConfig(mode, 4, 4)
        assert torch.equal(m(b, rc).logits, m2(b, rc).logits)


@pytest.mark.parametrize("mode", ["token", "pool"])
def test_tied_causal_and_incremental_equivalence(mode):
    torch.manual_seed(1)
    m = CSRModel(tied_tiny()).double().eval()
    ids = (np.arange(48) * 11) % 90 + 3
    starts = [0, 12, 30]
    b = single_doc_batch(ids, starts)
    rc = RoutingConfig(mode, 4, 4)
    full = m(b, rc).logits[0]
    st = m.new_state()
    from csr.segment import starts_to_arrays
    _, cpos = starts_to_arrays(len(ids), starts)
    for t in range(len(ids)):
        lg, _ = m.step(int(ids[t]), t, int(cpos[t]), rc, st)
        assert torch.allclose(lg, full[t], atol=1e-9)
    ids2 = ids.copy()
    ids2[30:] = 5  # future tokens changed -> earlier logits unchanged
    full2 = m(single_doc_batch(ids2, starts), rc).logits[0]
    assert torch.allclose(full[:30], full2[:30], atol=1e-12)


def test_tied_all_conditions_forward():
    from csr.config import CONDITIONS
    torch.manual_seed(0)
    m = CSRModel(tied_tiny()).double()
    ids = (np.arange(64) * 5) % 90 + 3
    b = single_doc_batch(ids, [0, 20, 44])
    for c, spec in CONDITIONS.items():
        rc = RoutingConfig(spec["routing.mode"], 4, 4)
        out = m(b, rc, need_chunk_logits=spec["semantic.enabled"])
        assert torch.isfinite(out.logits).all()
        con = out.routing[0]["constrained"]
        assert (con.sum() == 0) == (spec["routing.mode"] == "token")


def test_old_configs_unchanged_by_tie_flag():
    d = ModelConfig().to_dict()
    assert "tie_embeddings" not in d and ModelConfig.from_dict(d).tie_embeddings is False
    old = json.loads((ROOT / "configs" / "smoke-small-v1" / "condition_A.json").read_text())["model"]
    assert "tie_embeddings" not in old
    m = CSRModel(ModelConfig.from_dict(old))
    assert m.lm_head.weight is not m.tok_emb.weight


# ------------------------------------------------------------------ parameter counts and matched init
def test_rep_parameter_counts_exact():
    from rep_make_configs import ACTIVE, EMB, TOTAL
    c = rep_model_cfg()
    d, V, E, k = c.d_model, c.vocab_size, c.n_experts, c.top_k
    nm = len(c.moe_layers)
    analytic = V * d + c.n_layer * 4 * d * d + c.n_layer * 2 * d + d + (c.n_layer - nm) * 3 * d * c.dense_ffn \
        + nm * (d * E + E * 3 * d * c.expert_ffn + 3 * d * c.shared_ffn)
    m = CSRModel(c)
    cp = count_parameters(m)
    assert analytic == cp["total"] == TOTAL == 28_873_920
    assert cp["active_per_token"] == ACTIVE == analytic - nm * (E - k) * 3 * d * c.expert_ffn == 14_423_232
    assert 20e6 <= TOTAL <= 30e6 and (TOTAL - EMB) / TOTAL >= 0.5
    assert routing_pool(c) == 8


def routing_pool(c):
    return min(c.n_experts, 2 * c.top_k)


def test_rep_conditions_identical_initialization_within_seed_and_differ_across_seeds():
    from csr.config import CONDITIONS, build_condition_config
    from csr.train_lib import build_model
    from rep_make_configs import model_block
    sds = {}
    for seed in (0, 1):
        base = {"model": model_block(), "init": {"seed": seed, "type": "random"}, "routing": {}, "segmentation": {},
                "semantic": {}}
        for c in "AF":
            cfg = build_condition_config(base, c)
            sds[(seed, c)] = build_model(cfg, torch.device("cpu")).state_dict()
    a, f = sds[(0, "A")], sds[(0, "F")]
    assert list(a) == list(f) and all(torch.equal(a[k], f[k]) for k in a)
    assert not torch.equal(sds[(0, "A")]["tok_emb.weight"], sds[(1, "A")]["tok_emb.weight"])
    assert len(CONDITIONS) == 6


# ------------------------------------------------------------------ F-v3 random control
def _policy(h=0.3):
    from csr.rep import END_EDGES, HazardPolicyV3
    return HazardPolicyV3([0, 300, 600, 900, 2000], list(END_EDGES), 8, 64, 256, np.full((4, 5, 24), h).tolist())


def test_v3_is_content_free_and_obeys_state_machine():
    import inspect
    from csr.rep import hazard_starts_v3
    assert list(inspect.signature(hazard_starts_v3).parameters) == ["L", "ends", "policy", "doc_key", "seed", "decide"]
    rng = np.random.default_rng(0)
    pol = _policy()
    for trial in range(200):
        L = int(rng.integers(20, 1024))
        ends = np.unique(rng.integers(3, L, size=max(1, L // 15)))
        st = hazard_starts_v3(L, ends, pol, f"doc{trial}", 5)
        assert st[0] == 0 and all(b > a for a, b in zip(st, st[1:])) and st[-1] < L
        ls = np.diff(st + [L])
        assert ls.max() <= 256
        eset = set((ends + 1).tolist())
        for x, ln in zip(st[1:], ls[:-1]):
            assert x in eset or ln == 256 or (x - 256) in st  # closures at sentence ends or forced at 256
        assert st == hazard_starts_v3(L, ends, pol, f"doc{trial}", 5)  # deterministic
    a = [hazard_starts_v3(900, np.arange(10, 900, 9), pol, "k", s) for s in range(5)]
    assert len({tuple(x) for x in a}) > 1


def test_v3_replay_reproduces_semantic_boundaries():
    from conftest import fake_embed, topic_doc, fake_detector
    from csr.rep import replay_v3
    from csr.segment import SemanticParams, semantic_starts
    det, pol = fake_detector(), _policy()
    rng = np.random.default_rng(3)
    p = SemanticParams(0.5, 64, 256)
    for _ in range(30):
        ids = topic_doc(rng, n_sent_per_topic=tuple(int(x) for x in rng.integers(4, 30, 3)))
        e = det.sentence_ends(ids)
        x = fake_embed([ids[s:t + 1] for s, t in zip(np.r_[0, e[:-1] + 1], e)])
        st = semantic_starts(len(ids), e, x, p)
        rep, opp = replay_v3(len(ids), e, st, pol)
        assert rep == st


def test_v3_acceptance_criteria():
    from csr.rep import ACCEPT_V3, compare_v3, describe_v3
    Ls = [700, 300, 1000, 450] * 50
    st = [[0, 128, 300, 512][:1 + (L > 200) + (L > 400) + (L > 600)] for L in Ls]
    d = describe_v3(Ls, st)
    same = compare_v3(d, describe_v3(Ls, st))
    assert same["all_pass"] and all(same[k] == 0 for k in ACCEPT_V3)
    shifted = [[0] + [x + 40 for x in s[1:]] for s in st]
    c = compare_v3(d, describe_v3(Ls, shifted))
    assert not c["all_pass"] and not c["pass"]["nonfinal_mean_abs"]
    assert set(ACCEPT_V3) == {"chunk_count_rel", "nonfinal_mean_abs", "final_mean_abs", "eligible_frac_abs_pp", "ks_nonfinal"}
    assert ACCEPT_V3 == {"chunk_count_rel": 0.01, "nonfinal_mean_abs": 0.5, "final_mean_abs": 1.0,
                         "eligible_frac_abs_pp": 0.5, "ks_nonfinal": 0.02}


def test_v3_fit_matches_hazard_generated_D():
    """If D itself is generated by a v3 policy, the fit + realized comparison passes on a large synthetic set."""
    from csr.rep import compare_v3, describe_v3, fit_hazard_v3, hazard_starts_v3, simulate_v3
    rng = np.random.default_rng(0)
    true = _policy(0.0)
    h = np.asarray(true.hazard)
    h[:, :, :] = 0.15
    h[:, 0, :] = 0.45  # higher closure close to the document end
    true.hazard = h.tolist()
    docs = []
    for i in range(1500):
        L = int(rng.integers(100, 1024))
        docs.append((L, np.unique(rng.integers(3, L, size=L // 18)), f"d{i}"))
    D = [hazard_starts_v3(L, e, true, k, 77) for L, e, k in docs]
    pol, hist, mism = fit_hazard_v3(docs, D, true.strata_edges, iters=2, fit_seeds=(1,), select_seeds=(2,))
    assert mism == 0
    c = compare_v3(describe_v3([d[0] for d in docs], D), describe_v3([d[0] for d in docs], simulate_v3(docs, pol, 9)))
    assert c["chunk_count_rel"] < 0.03 and c["ks_nonfinal"] < 0.05


# ------------------------------------------------------------------ data helpers
def test_dedup_against_reference():
    from csr.rep import dedup_against_reference
    words = [f"w{i}" for i in range(400)]
    ref = [" ".join(words)]
    near = " ".join(words[:-1] + ["changed"])
    uniq = " ".join(f"u{i}" for i in range(300))
    ids = ["b", "a", "c", "d"]
    texts = [near, "  " + ref[0].upper() + " ", uniq, uniq.replace("u1 ", "U1  ")]
    keep, rep = dedup_against_reference(ref, ids, texts, processes=2)
    assert keep.tolist() == [False, False, True, False]
    assert rep["exact_vs_reference"] == 1 and rep["near_vs_reference"] == 1 and rep["exact_within_new"] == 1
    assert rep["n_new_kept"] == 1


def test_rep_split_rule_deterministic_and_proportions():
    from csr.rep import REP_SPLITS, rep_split_of
    names = [rep_split_of(f"<urn:uuid:{i}>") for i in range(40000)]
    assert names == [rep_split_of(f"<urn:uuid:{i}>") for i in range(40000)]
    for s, (lo, hi) in REP_SPLITS.items():
        assert abs(names.count(s) / len(names) - (hi - lo)) < 0.01


def test_rep_row_groups_exclude_old():
    from csr.rep import OLD_ROW_GROUPS, choose_rep_row_groups
    rg = choose_rep_row_groups(1000)
    assert len(rg) == 240 == len(set(rg)) and not set(rg) & set(OLD_ROW_GROUPS) and rg == choose_rep_row_groups(1000)


def test_embed_docs_batched_same_texts_as_reference_path():
    from csr.artifacts import load_lm_tokenizer
    from csr.data import TokenSplit
    from csr.rep import embed_docs_batched
    from csr.segcache import embed_doc_sentences, make_detector

    class FakeEnc:
        dim = 8

        def embed(self, texts):
            return np.array([[hash(t) % 997 + j for j in range(8)] for t in texts], np.float32)
    tok = load_lm_tokenizer()
    det = make_detector(tok)
    sp = TokenSplit.load(ROOT / "artifacts" / "data" / "fwe-v1", "calib")
    docs = [sp.doc(i) for i in range(25)]
    a = embed_doc_sentences(tok, det, FakeEnc(), docs)
    b = embed_docs_batched(tok, det, FakeEnc(), docs)
    for (e1, x1), (e2, x2) in zip(a, b):
        assert np.array_equal(e1, e2) and np.array_equal(x1, x2)


# ------------------------------------------------------------------ calibration selection rules
def test_tau_effective_counts_and_selection():
    from rep_calibrate_tau import TAU_R_GRID, TAU_S_GRID, eff_counts, select
    n = 50
    assert torch.allclose(eff_counts(torch.zeros(n, n), 0.1), torch.full((n,), n - 1.0))
    S = torch.eye(n) * 0 + torch.randn(n, n)
    assert (eff_counts(S, 1e-4) < eff_counts(S, 10.0)).all()
    q = {t: v for t, v in zip(TAU_S_GRID, [1.5, 3.0, 9.0, 30.0, 60.0, 90.0])}
    r = {t: 200.0 / (1 + i) for i, t in enumerate(TAU_R_GRID)}
    r[TAU_R_GRID[0]] = 8.5
    ts, tr, gap, ok = select(q, {t: r for t in TAU_S_GRID})
    assert ts == 0.05 and tr == TAU_R_GRID[0] and ok and abs(gap - 0.5 / 9) < 1e-12
    r2 = {t: 100.0 for t in TAU_R_GRID}
    assert select(q, {t: r2 for t in TAU_S_GRID})[3] is False  # uninformative endpoint -> STOP


def test_lambda_selection_rule():
    from rep_select_lambda import GRID, select_lambda

    def s(ratio, sem0, sem1, lm, collapse=False, status="COMPLETE"):
        return {"status": status, "finite": True, "median_ratio_final20": ratio, "sem_early20_mean": sem0,
                "sem_final20_mean": sem1, "lm_final20_mean": lm, "collapse": collapse}
    summ = {GRID[0]: s(0.01, 3.0, 3.1, 5.00), GRID[1]: s(0.03, 3.0, 2.9, 5.00), GRID[2]: s(0.08, 3.0, 2.7, 5.01),
            GRID[3]: s(0.15, 3.0, 2.5, 5.02), GRID[4]: s(0.4, 3.0, 2.0, 5.2)}
    chosen, crit = select_lambda(summ)
    assert chosen == GRID[2] and crit[GRID[2]]["all_pass"] and not crit[GRID[1]]["ratio_in_band"]
    summ[GRID[2]] = s(0.08, 3.0, 2.7, 5.05)  # +0.05 vs next smaller -> fails, next candidate considered
    chosen, crit = select_lambda(summ)
    assert not crit[GRID[2]]["lm_vs_next_smaller"] and chosen == GRID[3]  # GRID[3] vs GRID[2]: -0.03 within tol
    summ2 = {g: s(0.01, 3.0, 3.1, 5.0) for g in GRID}
    assert select_lambda(summ2)[0] is None
    summ3 = {g: s(0.1, 3.0, 2.0, 5.0, collapse=True) for g in GRID}
    assert select_lambda(summ3)[0] is None


# ------------------------------------------------------------------ accounting, checkpoint, eval pairing (rep data)
needs_rep = pytest.mark.skipif(not REP_CFG, reason="rep-v1 configs not generated yet")


@needs_rep
def test_rep_100M_token_accounting_identical_across_conditions():
    from csr.config import build_condition_config, load_json
    from csr.train_lib import valid_tokens
    from train import plan
    base = load_json(REP_CFG[0])
    plans = {}
    for c in "AEF":
        cfg = build_condition_config(base, c, {"semantic.lambda": 1e-4, "semantic.tau_s": 0.1, "semantic.tau_r": 1e-3})
        d, tr, steps, total = plan(cfg)
        plans[c] = (steps, total)
        assert total >= 100_000_000 and total - valid_tokens(steps[-1], tr) < 100_000_000
        assert all(len(s) == 32 for s in steps)
    assert plans["A"] == plans["E"] == plans["F"]


@needs_rep
def test_rep_checkpoint_roundtrip_and_eval_pairing(tmp_path):
    from csr.config import build_condition_config, load_json
    from csr.data import TokenSplit
    from csr.train_lib import BatchSource, build_model, evaluate
    from train import load_seg
    base = load_json(REP_CFG[0])
    tiny = {"model.n_layer": 2, "model.moe_layers": [1], "model.d_model": 32, "model.n_head": 4, "model.dense_ffn": 48,
            "model.expert_ffn": 16, "model.shared_ffn": 24, "model.n_experts": 8, "model.top_k": 2, "routing.pool_size": 4}
    res = {}
    for c in "AD":
        cfg = build_condition_config(base, c, tiny)
        m = build_model(cfg, torch.device("cpu")).eval()
        p = tmp_path / f"{c}.pt"
        torch.save({"model": m.state_dict(), "config": cfg}, p)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        m2 = CSRModel(ModelConfig.from_dict(sd["config"]["model"]))
        m2.load_state_dict(sd["model"])
        m2.eval()
        assert m2.lm_head.weight is m2.tok_emb.weight
        sp = TokenSplit.load(ROOT / cfg["data"]["dir"], "val")
        src = BatchSource(sp, 1024, load_seg(cfg, "val"))
        r1 = evaluate(m, src, cfg, torch.device("cpu"), 4, 4)
        r2 = evaluate(m2, src, cfg, torch.device("cpu"), 4, 4)
        assert r1["ce_nats_per_token"] == r2["ce_nats_per_token"] and r1["doc_nll"] == r2["doc_nll"]
        res[c] = r1
    assert res["A"]["doc_n"] == res["D"]["doc_n"] and res["A"]["n_valid_tokens"] == res["D"]["n_valid_tokens"]
    assert res["D"]["pool_constrained_coverage"] > 0.5 and res["A"]["pool_constrained_coverage"] == 0
