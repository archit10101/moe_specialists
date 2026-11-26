"""smoke-small-v1 (ENGINEERING SMOKE TESTS ONLY): exact parameter counts, identical models across A-F, exactly
k=2 routed experts per token, pools of m=4, and the frozen full-size v2 design left untouched."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
SMOKE = ROOT / "configs" / "smoke-small-v1"
V2 = ROOT / "configs" / "v2"
pytestmark = pytest.mark.skipif(not SMOKE.exists(), reason="smoke config not generated")

from csr.config import CONDITIONS, check_matched, flatten, manifest  # noqa: E402
from csr.model import CSRModel, ModelConfig, count_parameters  # noqa: E402

SMOKE_TOTAL, SMOKE_ACTIVE = 15_691_392, 13_971_072
V2_TOTAL, V2_ACTIVE = 100_670_208, 37_952_256


def load(dirpath, c):
    return json.loads((dirpath / f"condition_{c}.json").read_text())


def analytic(m: ModelConfig):
    d, V = m.d_model, m.vocab_size
    emb = 2 * V * d  # untied input embedding + LM head
    attn = m.n_layer * (3 * d * d + d * d)
    norms = m.n_layer * 2 * d + d
    dense = (m.n_layer - len(m.moe_layers)) * 3 * d * m.dense_ffn
    expert = 3 * d * m.expert_ffn
    moe = len(m.moe_layers) * (d * m.n_experts + m.n_experts * expert + 3 * d * m.shared_ffn)
    total = emb + attn + norms + dense + moe
    active = total - len(m.moe_layers) * (m.n_experts - m.top_k) * expert
    return total, active


def test_smoke_shape_and_exact_parameter_counts():
    cfg = load(SMOKE, "A")
    m = ModelConfig.from_dict(cfg["model"])
    assert (m.d_model, m.n_layer, m.moe_layers, m.n_head, m.n_experts, m.top_k, m.expert_ffn, m.shared_ffn,
            m.dense_ffn, m.max_seq_len) == (128, 6, (1, 2, 3, 4, 5), 8, 16, 2, 64, 128, 512, 1024)
    assert cfg["routing"]["pool_size"] == 4 and cfg["routing"]["warmup"] == 16 and cfg["data"]["seq_len"] == 1024
    model = CSRModel(m)
    cp = count_parameters(model)
    assert analytic(m) == (SMOKE_TOTAL, SMOKE_ACTIVE)
    assert (cp["total"], cp["active_per_token"]) == (SMOKE_TOTAL, SMOKE_ACTIVE)
    assert cfg["result_label"].startswith("ENGINEERING SMOKE TEST ONLY")


def test_v2_full_size_design_untouched():
    v2 = {c: load(V2, c) for c in CONDITIONS}
    m = ModelConfig.from_dict(v2["A"]["model"])
    assert (m.n_experts, m.top_k, v2["A"]["routing"]["pool_size"], m.d_model, m.n_layer) == (64, 6, 12, 256, 9)
    assert analytic(m) == (V2_TOTAL, V2_ACTIVE)
    recorded = json.loads((ROOT / "EXPERIMENT_MANIFEST.json").read_text())["shared_fields_sha256"]
    assert manifest(v2, None, None)["shared_fields_sha256"] == recorded
    assert "result_label" not in v2["A"]
    smoke = json.loads((SMOKE / "base.json").read_text())
    from csr.guard import sha256_file
    assert smoke["derived_from"]["sha256"] == sha256_file(V2 / "base.json")


def test_smoke_conditions_matched_and_models_identical():
    cfgs = {c: load(SMOKE, c) for c in CONDITIONS}
    assert check_matched(cfgs)
    assert len({json.dumps(c["model"], sort_keys=True) for c in cfgs.values()}) == 1
    from csr.train_lib import build_model
    sds = {c: build_model(cfg, torch.device("cpu")).state_dict() for c, cfg in cfgs.items()}
    names = {c: [(k, tuple(v.shape)) for k, v in sd.items()] for c, sd in sds.items()}
    assert all(n == names["A"] for n in names.values())
    for c in "BCDEF":
        assert all(torch.equal(sds["A"][k], sds[c][k]) for k in sds["A"])  # identical seeded initialization


def test_routing_validation_rejects_inconsistent_pool():
    from csr.train_lib import routing_from_cfg
    cfg = load(SMOKE, "B")
    assert routing_from_cfg(cfg).pool_size == 4
    for bad in (12, 1, 3, 17):
        with pytest.raises((AssertionError, ValueError)):
            routing_from_cfg({**cfg, "routing": {**cfg["routing"], "pool_size": bad}})


@pytest.mark.parametrize("cond", list(CONDITIONS))
def test_exactly_two_routed_experts_per_token_real_data(cond):
    from csr.data import TokenSplit, to_torch
    from csr.train_lib import BatchSource, build_model, forward_losses
    from train import load_seg
    cfg = load(SMOKE, cond)
    for k, v in (("tau_s", 0.1), ("tau_r", 0.01), ("lambda", 0.1)):  # placeholders for the forward check only
        if cfg["semantic"][k] is None:
            cfg["semantic"][k] = v
    sp = TokenSplit.load(ROOT / cfg["data"]["dir"], "val")
    seg = load_seg(cfg, "val")
    src = BatchSource(sp, cfg["data"]["seq_len"], seg)
    tb = to_torch(src.batch(src.rows[:2]), torch.device("cpu"))
    model = build_model(cfg, torch.device("cpu")).eval()
    with torch.no_grad():
        parts, out = forward_losses(model, tb, cfg, seg)
    nonpad = tb["doc_ids"] >= 0
    assert len(out.routing) == 5
    for r in out.routing:
        sel = r["topi"][nonpad]
        assert sel.shape[-1] == 2
        assert (sel[:, 0] != sel[:, 1]).all()  # two DISTINCT routed experts on every token
        assert ((sel >= 0) & (sel < 16)).all()
        if r["chunk_logits"] is not None and cfg["routing"]["mode"] == "pool":
            pools = r["chunk_logits"].topk(4, -1).indices
            con = r["constrained"]
            rows = out.layout.token_row[con]
            assert torch.isin(r["topi"][con], pools[rows]).all() if con.any() else True
            for t_sel, row in zip(r["topi"][con][:200], rows[:200]):
                assert set(t_sel.tolist()) <= set(pools[row].tolist())
    for blk in model.moe_blocks():  # shared expert path present in every MoE layer
        assert blk.ffn.shared is not None and blk.ffn.shared.w1.weight.shape == (128, 128)
    cov = parts.coverage_constrained / parts.n_valid
    assert (cov == 0) if cond == "A" else (cov > 0.5)
    assert np.isfinite(float(parts.lm_sum)) and abs(float(parts.lm_sum) / parts.n_valid - np.log(50304)) < 0.5
