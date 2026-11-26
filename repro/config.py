"""Single source of truth: every size, hyperparameter and experiment lives here.

Read this file top to bottom to know exactly what the pipeline runs.

  * ``Profile``    - global sizes. ``full`` is the reproduction, ``smoke`` is a tiny CPU-sized copy
                     that exercises every code path in a few minutes (used by tests, laptops, cloud).
  * ``EXPERIMENTS`` - the registry. One entry per experiment; each has a kind (base / bank / router),
                     the config overrides that define it, its seeds, its tier (core / ablation), and
                     which section of docs/source_history.md it reproduces.

Run ids are ``<experiment name>/s<seed>`` and live in ``$REPRO_WORK/runs/<experiment name>/s<seed>/``.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parents[1]))


def work_dir(profile: str) -> Path:
    """Outputs (data, groups, runs, results) for a profile. Override with $REPRO_WORK."""
    base = Path(os.environ.get("REPRO_WORK", ROOT / "work"))
    return base if profile == "full" else base / profile


# =============================================================================================
# Component configs
# =============================================================================================

@dataclass(frozen=True)
class DataCfg:
    source: str = "m2d2_arxiv"          # m2d2_arxiv | jsonl | synthetic
    domains: tuple = ("cs", "math", "physics", "q-bio", "stat")   # 5 arXiv L1 domains (doc §11)
    tokenizer: str = "gpt2"             # gpt2 (tiktoken) | bytes
    doc_tokens: int = 2048              # M2D2 has no paper boundaries: a "paper" = run of consecutive
                                        # paragraphs of about this many tokens (docs/RECONSTRUCTION.md)
    # tokens PER DOMAIN in each paper-disjoint split, filled in this order (contiguous ranges)
    split_tokens: tuple = (("test", 1_000_000), ("val", 1_000_000), ("calib", 1_000_000),
                           ("bank", 10_000_000), ("base", 49_200_000))
    jsonl_path: str = ""                # for source=jsonl: lines of {"text": ..., "domain": ...}


@dataclass(frozen=True)
class ModelCfg:
    vocab_size: int = 50304             # GPT-2 50257 padded to a multiple of 64
    ctx: int = 1024
    n_layer: int = 12
    n_head: int = 6
    d: int = 384
    n_exp: int = 32
    top_k: int = 4
    expert_hidden: int = 128            # SwiGLU experts; with the shared path -> ~104.8M params
    shared_hidden: int = 1536           # (doc: 103.6M; exact split unknown)


@dataclass(frozen=True)
class TrainCfg:
    tokens: int = 245_032_457           # doc §2.2
    batch_tokens: int = 65_536          # 64 x 1024 per optimizer step
    micro_bs: int = 16                  # windows per micro-batch (grad accumulation fills the step)
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    clip: float = 1.0
    switch_coef: float = 0.01           # doc §19 loss: L_LM + 0.01 L_Switch + 0.001 L_z
    z_coef: float = 0.001
    save_fracs: tuple = ()              # extra weight snapshots, e.g. (0.1, 0.25, 0.5)
    log_every: int = 25


@dataclass(frozen=True)
class EDLCfg:
    """Expert Divergence Learning (doc §3). groups=None disables it."""
    groups: str | None = None           # group table: domain | domain_shuf | sem_sharp | sem_sharp_shuf ...
    beta: float = 5e-3                  # fixed strength (explicit EDL)
    target_ratio: float | None = None   # if set: auto-calibrate beta so |grad EDL| = ratio * |grad LM|
    layers: str = "all"                 # all | upper
    min_mass: float = 32.0              # a group needs >= this many (weighted) tokens in a micro-batch


@dataclass(frozen=True)
class BankCfg:
    """Residual specialist bank on a frozen base (doc §23). The number of specialists is the number of
    clusters in ``groups`` (token / dense routing use ``groups`` only to define that count)."""
    base: str = "base.token4"           # experiment providing the frozen base
    base_seed: int = 0
    base_ckpt: str = "final"            # final | frac0.1 | frac0.25 | frac0.5
    routing: str = "oracle"             # oracle (group table) | token (learned top-1) | dense (always on)
    groups: str = "sem"                 # oracle group table (hard assignment per paper)
    width: str = "spec"                 # spec: every specialist has spec_hidden units
                                        # split: total_hidden // K each   total: one MLP of total_hidden
    spec_hidden: int = 256              # 384 -> 256 -> 384; 16 specialists = 3,145,728 params (doc §23)
    total_hidden: int = 4096            # 16 x 256
    tokens: int = 50_000_000
    lr: float = 1e-3
    balance_coef: float = 0.01          # token router only


@dataclass(frozen=True)
class RouterCfg:
    """Causal current-context router over a frozen bank (doc §27)."""
    bank: str = "bank.sem"
    bank_seed: int = 0
    prior_groups: str = "sem"           # semantic prior used by the training-only teacher
    inputs: str = "both"                # both | token | context
    ctx_window: int = 16
    arch: str = "gru"                   # gru | mlp
    hidden: int = 64                    # Linear(768->64) -> GRU(64) -> 16 logits = 75,152 params
    util_temp: float = 2.0              # teacher utility temperature, in units of utility std (2*s_u)
    sem_weight: float = 0.5             # teacher exponent on the semantic prior
    w_teacher: float = 1.0              # KL(teacher || router)
    w_ce: float = 0.01                  # expected-CE term (doc's "distillation weight 0.01"; RECONSTRUCTION.md)
    tokens: int = 50_000_000
    lr: float = 3e-3


@dataclass(frozen=True)
class ClusterCfg:
    """Label-free paper clusters. Tables built: sem, sem_sharp, sem_shuf, sem_sharp_shuf (main embedder,
    main_k), sem_k{K} / sem_k{K}_shuf for alt_ks, one table per extra embedder (main_k), and
    domain / domain_shuf from the data labels."""
    embedder: str = "e5"                # e5 | minilm | tfidf
    main_k: int = 16
    alt_ks: tuple = (4, 8, 32)          # ablation B5
    extra_embedders: tuple = ("minilm", "tfidf")   # ablation B7
    fit_docs: int = 60_000              # documents used to fit centroids
    soft_temp: float = 0.05             # soft posterior = softmax(cos / temp)
    sharp_temp: float = 0.02            # "sharpened" soft labels used by Semantic C6
    sinkhorn_iters: int = 50
    kmeans_iters: int = 30
    embed_tokens: int = 512             # first N tokens of a paper are embedded

    def tables(self) -> list[str]:
        t = ["domain", "domain_shuf", "sem", "sem_sharp", "sem_shuf", "sem_sharp_shuf"]
        for k in self.alt_ks:
            t += [f"sem_k{k}", f"sem_k{k}_shuf"]
        return t + list(self.extra_embedders)


@dataclass(frozen=True)
class EvalCfg:
    val_tokens: int = 5_000_000         # CE on (up to) this many val tokens
    batch: int = 16
    probe_windows_per_group: int = 24   # windows per group for ablation metrics (FSI, interaction)
    fsi_experts_per_layer: int = 2      # enriched experts removed per layer for group-ablation FSI
    interaction_layers: tuple = ()      # () = all layers
    min_routed: int = 200               # cells with fewer routed probe tokens are skipped


@dataclass(frozen=True)
class Profile:
    name: str
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    bank: BankCfg
    router: RouterCfg
    cluster: ClusterCfg
    eval: EvalCfg


FULL = Profile("full", DataCfg(), ModelCfg(), TrainCfg(), BankCfg(), RouterCfg(), ClusterCfg(), EvalCfg())

# Tiny copy: same code paths, synthetic data, bytes tokenizer, tf-idf clusters. Minutes on a laptop CPU.
SMOKE = Profile(
    "smoke",
    DataCfg(source="synthetic", tokenizer="bytes", doc_tokens=600,
            split_tokens=(("test", 30_000), ("val", 30_000), ("calib", 30_000),
                          ("bank", 120_000), ("base", 400_000))),
    ModelCfg(vocab_size=320, ctx=128, n_layer=2, n_head=2, d=64, n_exp=8, top_k=2,
             expert_hidden=32, shared_hidden=128),
    TrainCfg(tokens=600_000, batch_tokens=4096, micro_bs=16, lr=3e-3, min_lr=3e-4, log_every=10),
    BankCfg(spec_hidden=32, total_hidden=128, tokens=200_000, lr=3e-3),
    RouterCfg(hidden=16, tokens=150_000, lr=5e-3),
    ClusterCfg(embedder="tfidf", main_k=4, alt_ks=(8,), extra_embedders=("tfidf",), fit_docs=5_000,
               embed_tokens=400),
    EvalCfg(val_tokens=60_000, batch=16, probe_windows_per_group=6, min_routed=20),
)

PROFILES = {"full": FULL, "smoke": SMOKE}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]


# =============================================================================================
# Experiment registry
# =============================================================================================

@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str                           # base | bank | router
    seeds: tuple
    tier: str                           # core | ablation
    doc: str                            # what it reproduces / asks
    edl: EDLCfg = EDLCfg()              # base only
    save_fracs: tuple = ()              # base only
    bank: dict = field(default_factory=dict)     # BankCfg overrides (bank only)
    router: dict = field(default_factory=dict)   # RouterCfg overrides (router only)


SEEDS4 = (0, 1, 2, 3)


def _bank(r: str, **kw) -> dict:
    """Bank overrides for the three §23 conditions: sem (oracle), shuf (shuffled oracle), tok (= G)."""
    base = {"sem": dict(routing="oracle", groups="sem"),
            "shuf": dict(routing="oracle", groups="sem_shuf"),
            "tok": dict(routing="token", groups="sem")}[r]
    for key in ("groups",):
        if key in kw and r == "shuf":
            kw[key] = kw[key] + "_shuf"
    return {**base, **kw}


EXPERIMENTS: list[Experiment] = [
    # ---------------- Stage 1: base models (doc §18 Phase 1) -------------------------------------
    Experiment("base.token4", "base", SEEDS4, "core", "§18 TOKEN-4 baseline",
               save_fracs=(0.1, 0.25, 0.5)),
    Experiment("base.edl", "base", SEEDS4, "core", "§3/§18 explicit-label EDL",
               edl=EDLCfg(groups="domain", beta=5e-3)),
    Experiment("base.sem_c6", "base", SEEDS4, "core", "§18 Semantic C6 (label-free clusters)",
               edl=EDLCfg(groups="sem_sharp", target_ratio=0.02)),
    Experiment("base.shuf_c6", "base", SEEDS4, "core", "§18 shuffled C6 (persistent random grouping)",
               edl=EDLCfg(groups="sem_sharp_shuf", target_ratio=0.02)),
    # ablations
    Experiment("abl.edl_beta1e-3", "base", (0, 1), "ablation", "B1 EDL strength (weaker)",
               edl=EDLCfg(groups="domain", beta=1e-3)),
    Experiment("abl.edl_beta2e-2", "base", (0, 1), "ablation", "B1 EDL strength (stronger)",
               edl=EDLCfg(groups="domain", beta=2e-2)),
    Experiment("abl.edl_upper", "base", (0, 1), "ablation", "B2 EDL on upper half of layers only",
               edl=EDLCfg(groups="domain", beta=5e-3, layers="upper")),
    Experiment("abl.edl_shuflabels", "base", (0, 1), "ablation",
               "B3 explicit EDL with paper-permuted domain labels",
               edl=EDLCfg(groups="domain_shuf", beta=5e-3)),

    # ---------------- Stage 2: residual banks on frozen token4 s0 (doc §23) -----------------------
    Experiment("bank.sem", "bank", SEEDS4, "core", "§23 A: semantic oracle", bank=_bank("sem")),
    Experiment("bank.shuf", "bank", SEEDS4, "core", "§23 B: shuffled oracle", bank=_bank("shuf")),
    Experiment("bank.tok", "bank", SEEDS4, "core", "§23 C: learned token router (= generic bank G)",
               bank=_bank("tok")),
    # B4: the "base must already know English" hypothesis - same banks on partially trained bases
    *[Experiment(f"abl.bank_{r}_at{int(f * 100)}", "bank", (0, 1), "ablation",
                 f"B4 {r} bank on the token4 checkpoint at {int(f * 100)}% of training",
                 bank=_bank(r, base_ckpt=f"frac{f}"))
      for f in (0.1, 0.25, 0.5) for r in ("sem", "shuf", "tok")],
    # B5: granularity; total bank width fixed (total_hidden split over K specialists)
    *[Experiment(f"abl.bank_{r}_k{k}", "bank", (0, 1), "ablation", f"B5 {r} bank with K={k}",
                 bank=_bank(r, groups=f"sem_k{k}", width="split"))
      for k in (4, 8, 32) for r in ("sem", "shuf", "tok")],
    # B6: is it routing or just capacity?
    Experiment("abl.bank_dense_compute", "bank", (0, 1), "ablation",
               "B6 always-on residual, compute-matched (one 256-unit MLP)",
               bank=dict(routing="dense", width="spec")),
    Experiment("abl.bank_dense_params", "bank", (0, 1), "ablation",
               "B6 always-on residual, parameter-matched (one 4096-unit MLP)",
               bank=dict(routing="dense", width="total")),
    # B7: cluster quality
    Experiment("abl.bank_minilm", "bank", (0, 1), "ablation", "B7 MiniLM clusters", bank=dict(groups="minilm")),
    Experiment("abl.bank_tfidf", "bank", (0, 1), "ablation", "B7 tf-idf clusters", bank=dict(groups="tfidf")),
    Experiment("abl.bank_domain", "bank", (0, 1), "ablation", "B7 explicit 5-domain oracle",
               bank=dict(groups="domain", width="split")),
    # B8: other base seeds
    *[Experiment(f"abl.bank_{r}_base{b}", "bank", (0,), "ablation", f"B8 {r} bank on token4 seed {b}",
                 bank=_bank(r, base_seed=b))
      for b in (1, 2, 3) for r in ("sem", "shuf", "tok")],

    # ---------------- Stage 3: causal current-context router (doc §27) ---------------------------
    Experiment("router.sem", "router", (0, 1, 2), "core", "§27 GRU router over the frozen semantic bank"),
    Experiment("router.shuf", "router", (0, 1, 2), "core", "§27 same recipe over the shuffled bank",
               router=dict(bank="bank.shuf", prior_groups="sem_shuf")),
    # B9: independent specialist-training seeds (doc §32.3 item 1)
    *[Experiment(f"abl.router_{r}_bank{s}", "router", (0,), "ablation",
                 f"B9 router on the {r} bank trained with seed {s}",
                 router=dict(bank=f"bank.{r}", prior_groups="sem" if r == "sem" else "sem_shuf", bank_seed=s))
      for s in (1, 2, 3) for r in ("sem", "shuf")],
    # B10: router design
    Experiment("abl.router_token_only", "router", (0,), "ablation", "B10 input: current token only",
               router=dict(inputs="token")),
    Experiment("abl.router_ctx_only", "router", (0,), "ablation", "B10 input: context mean only",
               router=dict(inputs="context")),
    Experiment("abl.router_mlp", "router", (0,), "ablation", "B10 MLP instead of GRU", router=dict(arch="mlp")),
    Experiment("abl.router_ctx4", "router", (0,), "ablation", "B10 context window 4", router=dict(ctx_window=4)),
    Experiment("abl.router_ctx64", "router", (0,), "ablation", "B10 context window 64",
               router=dict(ctx_window=64)),
    Experiment("abl.router_sem0", "router", (0,), "ablation", "B10 teacher without semantic prior",
               router=dict(sem_weight=0.0)),
    Experiment("abl.router_sem1", "router", (0,), "ablation", "B10 teacher with semantic weight 1",
               router=dict(sem_weight=1.0)),
    Experiment("abl.router_ce_only", "router", (0,), "ablation", "B10 expected-CE only (no teacher)",
               router=dict(w_teacher=0.0, w_ce=1.0)),
]

REGISTRY = {e.name: e for e in EXPERIMENTS}
assert len(REGISTRY) == len(EXPERIMENTS), "duplicate experiment names"


def get_experiment(name: str) -> Experiment:
    if name not in REGISTRY:
        raise SystemExit(f"unknown experiment {name!r}. Known: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def bank_cfg(profile: Profile, exp: Experiment) -> BankCfg:
    return replace(profile.bank, **exp.bank)


def router_cfg(profile: Profile, exp: Experiment) -> RouterCfg:
    return replace(profile.router, **exp.router)


def edl_cfg(profile: Profile, exp: Experiment) -> EDLCfg:
    return exp.edl


def needed_tables(profile: Profile, exp: Experiment) -> set[str]:
    """Group tables an experiment reads (used to skip experiments a profile cannot run)."""
    if exp.kind == "base":
        return {exp.edl.groups} if exp.edl.groups else set()
    if exp.kind == "bank":
        return {bank_cfg(profile, exp).groups}
    rc = router_cfg(profile, exp)
    return {rc.prior_groups} | needed_tables(profile, get_experiment(rc.bank))


def runnable(profile: Profile, exp: Experiment) -> bool:
    return needed_tables(profile, exp) <= set(profile.cluster.tables())


def to_dict(obj) -> dict:
    return dataclasses.asdict(obj)
