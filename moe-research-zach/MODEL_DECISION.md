# MODEL_DECISION.md

Decision date: 2026-10-31 (Stage A). Status: **decided for Stage A/B. The pilot initialization needs user approval (see §5).**

## 1. Preferred checkpoint: CMU-FLAME/FLAME-MoE-38M-100M. Rejected as not practical.

Verified facts (HF API, official repo configs, paper):

| Item | Finding |
|---|---|
| Immutable revision | HF `c2b9dcae2f6785463ca7fae813536e12c439b798` |
| Format | Megatron `torch.distributed.checkpoint`: `iter_0002121/__{0..7}_{0,1}.distcp` (16 shards), `.metadata` (2.3 MB, a pickle), `common.pt` (a pickle referencing `argparse.Namespace` and `megatron.*`), `metadata.json`. 1,421,422,379 bytes in total. |
| HF loadability | No `config.json`, tokenizer or modeling code. Cannot be loaded with `transformers`. |
| Original stack | Megatron-LM fork `yuzc19/Megatron-LM@cbaf684`, TransformerEngine `fc03478`, apex `c02c6c8`, trained with EP=8 on 32×H100 |
| Architecture | 9 layers, d=256, 16 heads, RoPE, SwiGLU, RMSNorm, untied embeddings, Pythia tokenizer; layer 0 dense (FFN 1368), layers 1–8 MoE with 64 experts (FFN 176), **top-6 routed + 2 shared** (shared FFN 352) |
| Routing | Softmax over all 64 experts, then top-6 on probabilities, with no renormalization (paper Eq. 1). Router in fp32. No capacity flag (dropless inferred, not verified). aux 0.01, z-loss 0.001. |

Reasons it is not practical:

1. **No validated loading path.** The task's baseline-equivalence gate (§14.3) needs the original checkpoint's
   logits. Producing those means running the Megatron fork plus TransformerEngine, which is a heavy compiled
   CUDA build and is not installable within the disk budget below. A hand-written DCP→PyTorch converter
   could not be validated against the original, and the task forbids treating such output as validated.
2. **Untrusted pickles.** `common.pt` and the DCP `.metadata` must be unpickled. That is equivalent to running
   remote code unless a separate audit (opcode inspection plus an allow-listed unpickler) is authorized.
3. **Disk.** The filesystem had 110.07e9 bytes free at the start (102.5 GiB). Under the conservative reading of
   the 100 GB reserve as 100 GiB, the whole project budget is about 2.5 GiB. 1.42 GB of checkpoint plus a
   CUDA PyTorch install (about 4–5 GB) does not fit.

## 2. Small pretrained single-GPU MoE alternatives investigated

| Candidate | Why rejected |
|---|---|
| ibm-granite/granite-3.0-1b-a400m-base (`d91cbed8`, native HF, 32 experts top-8, no shared experts) | 5,541,949,275 bytes of safetensors, which exceeds the disk budget. Six fine-tuned 1.3B checkpoints would not fit either. |
| allenai/OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, Mixtral family | 14–90 GB, so too large (sizes from general knowledge, not queried) |
| EMO checkpoints (allenai/Emo_1b14b_*) | 14B total and `trust_remote_code`. Prior art only. |
| google/switch-base-* | Encoder–decoder, not a causal LM |

None is appropriate under the constraints.

## 3. Selected family: from-scratch FLAME-38M-architecture-matched MoE, written in plain PyTorch

- Implementation: `src/csr/model.py`, our own code, inspectable. nanoMoE (`661fbcf`, MIT) is a design
  reference only and is not vendored, because its routing semantics (renormalized top-k, capacity dropping)
  differ from FLAME's.
- Architecture: identical to the FLAME-38M configs in §1. Vocab is 50,304 (Pythia's 50,277, padded to a
  multiple of 128, as Megatron does). Totals are about 100.7M parameters and about 37.8M active.
- Routing semantics (FLAME): fp32 router with no bias, softmax over all E=64 experts, top-k=6 by probability,
  gate = full-softmax probability with no renormalization. Fully **dropless** (no capacity factor). Two shared
  experts are implemented as one SwiGLU of width 352, which is how Megatron implements shared experts.
  Switch-style aux loss with coefficient 0.01 and z-loss with coefficient 0.001.
- Pool-constrained tokens (default `gate_norm="global"`): the original top-k is taken over the original
  global-softmax probabilities, restricted to experts in the pool. Gate values stay the **unrenormalized global
  probabilities**, as in FLAME. Only the set of eligible experts changes, never the gate scale. It reduces
  exactly to the baseline when m=E.
  - Rejected default: EMO-style renormalization inside the pool (`gate_norm="pool"`, still implemented).
    It would raise the gate mass of constrained tokens several-fold (≈0.09 vs ≈0.5 at a near-uniform router
    with E=64, k=6, m=12). That confounds B−A with a change in routed-output scale.
  - The phrase "original weight normalization inside the pool" in the task text is ambiguous. This
    interpretation is a decision for user review.
  - Each eligible token keeps exactly k=6 active routed experts plus the 2 shared experts.
- Tokenizer: `EleutherAI/pythia-12b@bb1e3e710cdf6b524461d543cfb5ba773f0a81b6`, `tokenizer.json` only (2.1 MB,
  Apache-2.0).
- Context length: 1024. FLAME used 2048; 1024 is chosen to cut attention cost. Documents are truncated to
  1023 tokens plus EOS and packed with block-diagonal document masks.
- Frozen semantic encoder (external, not part of the LM): `sentence-transformers/all-MiniLM-L6-v2@1110a243f`,
  Apache-2.0, 90.9 MB. It is loaded through native `transformers.BertModel` with mean pooling and L2
  normalization, and no `trust_remote_code`.

**Consequence:** this is **not** an experiment on a pretrained public checkpoint. Results describe either
from-scratch training of a small MoE or adaptation of our own small base model (§5). They do not say
anything about pretraining efficiency or emergent modularity at scale.

## 4. Trainable and frozen parameter sets (identical for all conditions A–F)

- Trainable: **all LM parameters**. That is `tok_emb.weight`, every `blocks.*` parameter (attention,
  RMSNorm, dense FFN, routers `blocks.{1..8}.ffn.router.weight`, experts `blocks.{1..8}.ffn.experts.{w1,w2,w3}`,
  shared experts `blocks.{1..8}.ffn.shared.*`), `norm_f.weight` and `lm_head.weight`. The exact list is in the
  manifest.
- Frozen: the external MiniLM encoder and both tokenizers. These are never handed to the optimizer, and a
  test checks that their parameters are unchanged.
- The CSR conditions add no parameters. The pool selector reuses each layer's existing router `W_r`.

## 5. Initialization: requires user approval before the pilot

Training from random initialization for the task's 3–5M tokens per condition would leave the routers almost
untrained. A no-training diagnostic on such a model is uninformative. **Proposal for the pilot, not authorized:**

1. Pretrain one shared base model with condition-A (token-level) routing on a separate FineWeb-Edu training
   stream. The size will be set from the measured Stage B throughput.
2. Initialize **every** condition A–F from that same base checkpoint and adapt it for an equal number of valid
   tokens.
3. Run the no-training diagnostic on the base checkpoint.

Known bias: the base was trained with token routing, so A has a home-regime advantage at equal adaptation
tokens. That is the same as fine-tuning any public token-routed checkpoint, and it is reported as a limitation.
The alternative for the user to choose is from-scratch training per condition.

Stage B work (smoke test, benchmark, probe pipeline check) uses a fixed-seed random initialization and makes
no scientific claim.

## 6. Unsupported alternatives (explicitly not done)

- Converting FLAME DCP checkpoints (§1). It could be revisited only with authorization for a pickle audit, more
  disk, and an acceptance criterion weaker than logit equivalence, such as a held-out loss sanity band. Any
  such result would be labeled UNVALIDATED.
- EMO checkpoints, `trust_remote_code`, Megatron, TransformerEngine.
