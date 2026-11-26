#!/usr/bin/env python
"""csr-semantic-edl-v1 CPU tests (prereg section 5), CUDA hidden. Writes artifacts/semantic-edl-v1/cpu_tests_<UTC>.json.
Usage: CUDA_VISIBLE_DEVICES= .venv/bin/python sem_tests.py [name prefixes]"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RES, TESTS = {}, []
TINY = {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
        "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False}
TRAIN = {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 1, "lr": 6e-4,
         "micro_batch_rows": 32, "min_lr": 6e-5, "rows_per_step": 64, "warmup_frac": 0.01, "weight_decay": 0.1,
         "z_coeff": 0.001}


def test(fn):
    TESTS.append(fn)
    return fn


@test
def t1_q_paper_constant_and_frozen():
    import numpy as np
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.sem_paths import SEM_TABLE, SEM_TABLE_SHA
    sp, steps, _ = main_stream(245_000_000)
    pq = PaperQ(SEM_TABLE, sp, SEM_TABLE_SHA)
    first = {}
    for di, p in enumerate(sp.papers):
        if p in first:
            assert np.array_equal(pq.q_of_doc[first[p]], pq.q_of_doc[di])
        else:
            first[p] = di
    from csr.fsp_lib import make_batch
    npb = make_batch(steps[5][:32], sp)
    qs = pq.qseq(npb)
    sid, gd = npb["seq_id"].reshape(-1), npb["gdoc"].reshape(-1)
    for s in range(qs.shape[0]):
        g = np.unique(gd[sid == s])
        assert len(g) == 1 and np.array_equal(qs[s], pq.q_of_doc[g[0]])
    assert pq.coverage > 0.97, pq.coverage
    for name in ("table_SEM_T05", "table_SHUF_SEM", "table_SHUF_SEM_T05"):
        t = json.loads((guard.ROOT / "artifacts/semantic-edl-v1/tables/tables.json").read_text())[name]
        assert guard.sha256_file(guard.ROOT / t["path"]) == t["sha256"]


@test
def t2_soft_loss_equals_explicit_edl_onehot():
    import torch
    from csr.fsp_model import edl_loss
    from csr.gc_lib import gc_edl_loss
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        S, E = 14, 32
        sid = torch.repeat_interleave(torch.arange(S), torch.randint(5, 40, (S,), generator=g))
        p = torch.softmax(torch.randn(len(sid), E, generator=g, dtype=torch.float64), -1)
        lab = torch.randint(0, 5, (S,), generator=g)
        lab[lab == 1] = 2
        nonpad = torch.rand(len(sid), generator=g) > 0.05
        a = edl_loss(p, lab[sid], sid, nonpad)
        b = gc_edl_loss(p, torch.nn.functional.one_hot(lab, 5).double(), sid, nonpad)
        assert abs(float(a) - float(b)) < 1e-9


@test
def t3_gradients_reach_routers_and_placement():
    import torch
    from csr.gc_lib import gc_edl_loss
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(n_layer=4, d_model=64, n_head=2, dense_ffn=128, n_experts=8, expert_ffn=32, shared_ffn=64,
                    vocab_size=512)
    m = LFRModel(cfg, 0)
    ids = torch.randint(0, 512, (4, 64), generator=torch.Generator().manual_seed(0))
    doc = torch.zeros_like(ids)
    doc[:, 32:] = 1
    sid = (torch.arange(4).view(-1, 1) * 2 + doc).reshape(-1)
    q = torch.softmax(torch.randn(8, 4, generator=torch.Generator().manual_seed(1)), -1)
    out = m({"input_ids": ids, "doc_ids": doc, "positions": torch.arange(64).expand(4, 64) % 32}, {"collect": True})
    routers = [n for n, _ in m.named_parameters() if n.endswith("router.weight")]
    assert len(routers) == 3
    sel = [1]  # MoE index 1 only (placement subset)
    loss = torch.stack([gc_edl_loss(out["records"][li]["p"].reshape(-1, 8).float(), q, sid,
                                    torch.ones_like(sid, dtype=torch.bool)) for li in sel]).mean()
    grads = torch.autograd.grad(loss, [dict(m.named_parameters())[r] for r in routers], allow_unused=True)
    norms = [0.0 if g_ is None else float(g_.norm()) for g_ in grads]
    assert norms[1] > 0 and norms[0] > 0 and norms[2] == 0.0, norms  # earlier layers via hidden states; later none


def _tiny_cfg(work, name, cond, milestones=None, diag=0):
    from csr.sem_paths import prereg_sha
    cfg = {"run_name": name, "seed": 3, "budget_tokens": 245_000_000, "prereg_sha256": prereg_sha(), "model": TINY,
           "cond": cond, "milestones": milestones or {}, "diag_every": diag, "train": TRAIN}
    p = Path(work) / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


def _go(cfgp, work, extra):
    env = {**os.environ, "MOE_RUNS_DIR": str(Path(work) / "runs"), "MOE_RESEARCH_APPROVE": "cpu-model",
           "CUDA_VISIBLE_DEVICES": ""}
    (Path(work) / "runs").mkdir(exist_ok=True)
    before = set((Path(work) / "runs").iterdir())
    r = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "sem_train.py", "--config", str(cfgp), "--execute"]
                       + extra, env=env, capture_output=True, text=True, cwd=guard.ROOT)
    new = sorted(set((Path(work) / "runs").iterdir()) - before)
    return r, (new[0] if new else None)


def _metrics(d):
    return {x["step"]: x for x in map(json.loads, (Path(d) / "metrics.jsonl").read_text().splitlines()) if "step" in x}


def _cleanup(work, dirs):
    import shutil
    from csr.sem_paths import TMP
    for d in dirs:
        if d is not None:
            shutil.rmtree(TMP / "slots" / Path(d).name, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


@test
def t4_token4_bit_identical_when_term_disabled():
    from csr.sem_paths import SEM_TABLE, SEM_TABLE_SHA, TMP
    work = Path(tempfile.mkdtemp(dir=TMP))
    c0 = _tiny_cfg(work, "t4a", {})
    c1 = _tiny_cfg(work, "t4b", {"paper_latent": {"table": str(SEM_TABLE.relative_to(guard.ROOT)),
                                                   "table_sha256": SEM_TABLE_SHA, "beta": 0.0}})
    r0, d0 = _go(c0, work, ["--max-steps", "3"])
    r1, d1 = _go(c1, work, ["--max-steps", "3"])
    assert r0.returncode == 75 and r1.returncode == 75, r1.stderr[-2000:]
    m0, m1 = _metrics(d0), _metrics(d1)
    for s in range(3):
        assert m0[s]["lm"] == m1[s]["lm"] and m0[s]["gn"] == m1[s]["gn"], (s, m0[s], m1[s])
    assert "gcedl" in m1[0] and m1[0]["gcedl"] > 0
    _cleanup(work, [d0, d1])


@test
def t5_shuffled_table_destroys_association():
    sc = json.loads((guard.ROOT / "artifacts/semantic-edl-v1/tables/tables.json").read_text())["shuffle_check"]
    assert abs(sc["hard_agreement"] - sc["chance_agreement"]) < 0.005 and sc["mi_nats_sem_vs_shuf"] < 0.01, sc


@test
def t6_per_expert_multi_reproduces_fle_domain():
    import numpy as np
    import torch
    from fle_evaluate import per_expert
    from csr.fsp_lib import Split
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.sem_lib import per_expert_multi
    cfg = LFRConfig(n_layer=3, d_model=64, n_head=2, dense_ffn=128, n_experts=8, expert_ffn=32, shared_ffn=64,
                    vocab_size=50304)
    m = LFRModel(cfg, 0).eval()
    cs = Split("calib_search")
    rd = eval_rows(cs, 3000, paper_half(cs, 0))
    rp = eval_rows(cs, 1500, paper_half(cs, 1))
    dev = torch.device("cpu")
    disc = evaluate(m, cs, rd, dev, per_token=True)["per_token"]
    a = per_expert(m, cs, rd, cs, rp, dev, disc)
    b = per_expert_multi(m, cs, rd, cs, rp, dev, disc, {"domain": (disc["dom"].astype(np.int64),
                                                                   lambda x: x["dom"], 5)})["domain"]
    assert a["interaction"] == b["interaction"] and a["kstar"] == b["kstar"], (a["interaction"][:3],
                                                                              b["interaction"][:3])


@test
def t7_exact_resume_cpu_with_layers_and_ramp_and_inference():
    import torch
    from csr.sem_paths import TMP
    work = Path(tempfile.mkdtemp(dir=TMP))
    t = json.loads((guard.ROOT / "artifacts/semantic-edl-v1/tables/tables.json").read_text())["table_SEM_T05"]
    cond = {"paper_latent": {"table": t["path"], "table_sha256": t["sha256"], "beta": 1e-2, "layers": [1],
                             "warmup_frac": 0.001}}
    c = _tiny_cfg(work, "r", cond, {"m4": 4}, diag=2)
    r, full = _go(c, work, ["--max-steps", "5"])
    assert r.returncode == 75, r.stdout[-2000:] + r.stderr[-2000:]
    r, part = _go(c, work, ["--max-steps", "2"])
    r2, _ = _go(c, work, ["--max-steps", "5", "--run-dir", str(part)])
    assert r2.returncode == 75, r2.stderr[-2000:]
    la, lb = _metrics(full), _metrics(part)
    for s in range(5):
        assert la[s]["lm"] == lb[s]["lm"] and la[s]["gcedl"] == lb[s]["gcedl"], (s, la[s], lb[s])
    diag = [json.loads(x)["diag"] for x in (full / "metrics.jsonl").read_text().splitlines() if "diag" in x]
    assert diag[0]["step"] == 0 and diag[0]["gn_aux"] == 0.0 or diag[0]["val_aux"] == 0.0  # ramp starts at 0
    assert any(d["gn_aux"] > 0 for d in diag[1:])
    from csr.lfr_model import LFRConfig, LFRModel
    ck = torch.load(full / "ckpt_m4" / "state.pt", map_location="cpu", weights_only=False)
    assert set(ck["model"]) == set(LFRModel(LFRConfig.from_dict(TINY), 0).state_dict())
    code = ("import sys,torch; sys.path.insert(0,'src')\n"
            "def hook(e,a):\n"
            "    if e=='open' and a and any(s in str(a[0]) for s in ('gc-edl-v1','semantic-edl-v1/tables','bge')):\n"
            "        raise PermissionError(str(a[0]))\n"
            "from csr.lfr_model import LFRConfig, LFRModel\n"
            f"ck=torch.load('{full}/ckpt_m4/state.pt',map_location='cpu',weights_only=False)\n"
            "sys.addaudithook(hook)\n"
            "m=LFRModel(LFRConfig.from_dict({**ck['model_cfg'],'teacher':False}),0); m.load_state_dict(ck['model'])\n"
            "ids=torch.randint(0,50304,(1,32)); o=m({'input_ids':ids,'doc_ids':torch.zeros_like(ids),"
            "'positions':torch.arange(32)[None]}); print('OK')\n")
    rr = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "-c", code], capture_output=True, text=True,
                        cwd=guard.ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert "OK" in rr.stdout, rr.stderr[-2000:]
    _cleanup(work, [full, part])


def main():
    guard.check_root()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA must be hidden"
    sel = sys.argv[1:]
    for fn in TESTS:
        if sel and not any(fn.__name__.startswith(s) for s in sel):
            continue
        t0 = time.time()
        try:
            fn()
            RES[fn.__name__] = {"ok": True, "s": round(time.time() - t0, 1)}
        except Exception:
            RES[fn.__name__] = {"ok": False, "err": traceback.format_exc()[-2500:]}
        print(fn.__name__, RES[fn.__name__]["ok"], flush=True)
    from csr.sem_paths import ART
    ok = all(v["ok"] for v in RES.values())
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "all_ok": ok, "selected": sel, "results": RES})
    print(json.dumps({"all_ok": ok, "out": str(out)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
