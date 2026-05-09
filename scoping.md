# Intra-block delta model — scoping doc

Companion to `design.md`. Captures the locked design decisions from the
discussion and proposes a concrete file layout, training schedule, and
milestone plan. References `intra_block_drift_findings.md` for the empirical
anchors and the lit-search verdict (closest prior work: EAGLE-1
arXiv:2401.15077; design hint sources: ControlNet, Flamingo, EAGLE-3, DiT).

---

## Goal

Train a small per-base-model "delta predictor" that, conditioned on a
recent backbone-actual hidden state `h_ref` (the most recent backbone
forward in the current block), emits `Δh` such that
`h_ref + Δh ≈ h_target_actual` at some later pass `i_target` in the
same block. At inference, the delta model produces predicted hiddens
for one or more future passes; a confidence head decides when to
rollback to a fresh backbone forward, whose output becomes the new
`h_ref`. Training pairs `(i_ref, i_target)` are sampled with
`1 ≤ i_ref < i_target ≤ 6` — max iteration capped at 6 to bound
storage. Success metric: **GSM8K accuracy within ε of full-backbone
Fast-dLLM**, not speculative-decoding acceptance rate.

---

## Locked design

**Architectural**
- Per-base-model training (LLaDA-8B and Dream-7B are *separate* runs;
  no transfer goal).
- Block size **32** (Fast-dLLM default).
- Δh emitted **at every block position**. CC positions trained with
  Δh ≈ 0 organically.
- `prev_emb` at non-revealed positions = the **mask-token embedding**
  (the same embedding the backbone would use at those positions).
- Prefix KV scope: **last layer only** (matches T3).
- `h_ref` conditioning input: **last-layer hidden only** for M1.
  Multi-layer fusion (EAGLE-3 hint, e.g. `{layer 1, mid, last}`)
  deferred to M2 as a candidate enhancement — drops collect storage
  ~3× and keeps the v0 model simpler. Reinstate fusion only if M1
  shows last-layer-only conditioning is the bottleneck.
- **Zero-init** the Δh output projection (ControlNet/Flamingo pattern).
  At step 0 the model = "reuse h_1 verbatim" baseline.
- Delta model size: **start at 2 transformer layers**, scale up if needed.
- **No iteration-index embedding** — `Δh` is deterministic given
  `h_ref` (which encodes the reveal state at `i_ref`) plus the
  target-pass reveal pattern (encoded in `prev_emb`); the integer
  pair `(i_ref, i_target)` carries no extra info.

**Data**
- Source prompts: **Nemotron-3** chat / math / code subsets.
- Generators: LLaDA-8B and Dream-7B run with Fast-dLLM v1 dual-cache
  parallel decoding; one cached dataset per generator.
- Per block, per pass `i ∈ {1..6}`: cache the **last-layer hidden**
  (block_len × d, fp16), the input revealed-token pattern, and the
  still-mask boolean. Same tensor doubles as `h_ref` source (when
  pass `i` is sampled as ref) and as the supervision target (when it
  is sampled as target).
- Prefix KV (last layer) cached as **8 last-32-token snapshots per
  sample** — one per block. Slicing rule: take the last 32 tokens
  of the full prefix at each block's pass-1 forward. For block `b ≥ 1`
  this is decoded block `b-1`'s 32 tokens; for block 0 it's the
  trailing 32 prompt tokens (chat-template tail). Bold compression:
  the delta model loses direct attention to the bulk of the prompt
  / earlier decoded blocks, but `h_ref` (the backbone's last-layer
  output, smeared by bidirectional attention) carries that context
  implicitly. Reinstate full prefix at M2 only if M1's per-block
  error breakdown shows block 0 (template-tail-only context) is the
  dominant failure mode.
- Reason this can't be deduped further (e.g., one prompt-KV snapshot
  reused across blocks): LLaDA / Dream are **bidirectional**, and
  Fast-dLLM v1 redoes a full-sequence forward at the start of every
  new block (`generate_with_dual_cache` and `generate_with_prefix_cache`
  in `v1/llada/generate.py:132,211`). The K/V at "the same" prompt
  token differs across blocks because the un-decoded suffix has
  changed. Each snapshot has to be the actual K/V the backbone
  produced at that block's pass-1.
- `max_iter = 6` is fixed for now; passes 7+ within a block are
  dropped to bound storage. Revisit if M2 shows late-pass coverage
  matters.

**Loss (day 1)**
```
L = λ_mse  · MSE(Δh_pred, Δh_target)                          [λ_mse  = 1.0]
  + λ_kl   · KL( p_predicted || p_actual )  at mask positions [λ_kl   = 0.1]
  + λ_conf · BCE( c_pred, c_label )                           [λ_conf = 0.1]

  Δh_target  = h_target_actual − h_ref_actual    (i_ref < i_target ≤ 6)
  p_actual   = softmax( lm_head( final_norm( h_target_actual         ) ) )
  p_predicted= softmax( lm_head( final_norm( h_ref_actual + Δh_pred ) ) )
  c_label    = mean_{p ∈ mask} shared_mass( p_actual[p], p_predicted[p] )
  c_pred     = sigmoid( scalar head on pooled features )
```
- KL evaluated at mask positions only (CC carries no signal).
- `c_label` detached from autograd on the prediction side — only `c_pred`
  trains; `Δh` is supervised by MSE+KL, not by the conf loss.
- No L2 on ‖Δh‖ at v0. Add only if overshoot is observed.

**Sampling**
- v0: **uniform** over `(i_ref, i_target)` pairs with
  `1 ≤ i_ref < i_target ≤ 6`, jointly with reveal-fraction.
  No curriculum.
- v0.1 (only if v0 plateaus): curriculum on **reveal-fraction**, easy
  → hard (high-reveal → low-reveal). Iteration-index curriculum
  abandoned (your h_2→h_3→all proposal subsumed by reveal-fraction
  curriculum).
- Always report metrics **binned by `(i_ref, i_target)` gap and by
  reveal-fraction**, never averaged alone.

**Inference / eval**
- Inference engine is built on **`generate_with_prefix_cache`**
  (`v1/llada/generate.py:132`), not `generate_with_dual_cache`.
  Reason: prefix-cache mode caches *only* the prefix KV at each
  block's pass-1 forward and recomputes the in-block forward at every
  later pass. Our delta model is a 1:1 swap for that in-block forward;
  there is no equivalent place to slot it into dual-cache mode (which
  also caches block-region KV via `replace_position`, and our delta
  model doesn't produce a block-region KV update).
- **Headline baseline**: vanilla `generate_with_prefix_cache` on the
  same prompts. Our hybrid replaces its in-block forward — speed
  win comes from the delta model being ≪ the backbone.
- **Stretch baseline**: `generate_with_dual_cache`. Closing the speed
  gap to dual-cache requires teaching the delta model to also output
  a block-region KV update — defer to M2.

---

## Architectures (three variants)

All three share:
- 2 transformer layers (default), `d_model` = backbone d_model, num heads
  matching backbone for KV-shape compatibility.
- Cross-attention into the cached **prefix KV** at every layer
  (last-layer KV from pass 1).
- Zero-init final projection to Δh.
- Scalar conf head: pool the final-layer features (mean-pool over
  block_len), linear → sigmoid.

### C — 2s-sequence concat (baseline)
- Concatenate `h_1` and `prev_emb` along the sequence axis →
  length `2·block_len`.
- Add learned positional embedding distinguishing the two halves.
- Feed through self-attention + prefix-KV cross-attention layers.
- Take the second half (length `block_len`) as features → Δh head.
- Simplest, no conditioning cleverness. If A/B don't beat this clearly,
  the conditioning style isn't doing work.

### B — AdaLN-Zero modulation
- `prev_emb` is the residual stream.
- `h_1` is pooled per-position to a feature vector and used to predict
  per-layer `(γ, β)` for the LayerNorms (DiT AdaLN-Zero recipe).
- Cross-attention into prefix KV at each layer.
- Param-efficient. Lit caveat: AdaLN-Zero won when conditioning was
  low-dim; high-dim banks usually favor cross-attn — included as the
  efficient alternative to A.

### A — Cross-attn anchor (primary)
- `prev_emb` is the residual stream (queries).
- `h_1` is exposed as a frozen K/V bank at every layer
  (gated cross-attention, Flamingo-style with tanh-α gate init at 0).
- Cross-attention into prefix KV separately, also at every layer.
- Lit's default for high-dim conditioning. Closest match to "h_1 is
  the anchor I should respect, prev_emb is the perturbation."

**Implementation order:** C → B → A (cheapest debug → richest model).

---

## Data pipeline

```
collect.py  (one-shot per base model)
  ├─ load Nemotron-3 chat/math/code prompts
  ├─ for each prompt:
  │    run base model + Fast-dLLM v1 (mode-agnostic — dual or
  │    prefix cache both produce the same pass-1 KV):
  │      per block b ∈ {0..7}:
  │        - capture last-32-token prefix KV (last layer)    [once/block]
  │        per pass i ∈ {1..6}:
  │          - capture h_i (last layer, block_len × d)       [each pass]
  │          - capture revealed-token pattern + mask vector  [each pass]
  └─ shard to disk (one shard per ~N prompts)

dataset.py
  ├─ yields one (block, i_ref, i_target) example per __getitem__
  │   with i_ref < i_target sampled jointly
  ├─ load that block's last-32 prefix KV snapshot directly
  ├─ on-the-fly: build prev_emb at i_target from revealed pattern
                 + base-model token embed (frozen) + mask-embed at masked positions
  └─ yields: (prefix_kv_last32, h_ref, prev_emb, mask,
              h_target_actual, h_ref_actual)
```

Storage estimate (M1, fusion dropped, last-32-token prefix snapshots):

```
per sample (8 blocks, 6 passes/block, d=4096, fp16, last layer):
  hidden cache:           8 × 6 × 32 × 4096 × 2 B            ≈ 12 MB
  prefix KV (8 × 32 tok): 8 × 32 × 4096 × 2 × 2 B            ≈  4 MB
  reveal/mask metadata                                       ~  negligible
  ─────────────────────────────────────────────────────────
  total:                                                     ≈ 16 MB / sample
```

5000 samples → ~80 GB; 10000 → ~160 GB. Both fit comfortably on a
single drive. M1 plan: **5000 samples**, with room to scale to 10000
at M2 without storage-strategy changes. Either of the deferred
enhancements (full-prefix snapshots, multi-layer fusion) costs
~5–10× on storage; revisit at M2 only if M1 caps the recipe.

---

## File / module layout

```
peft_project/tata/
  design.md                            (existing)
  scoping.md                           (this doc)
  delta_model/
    __init__.py
    data/
      collect_llada.py                 (cache builder, LLaDA backbone)
      collect_dream.py                 (cache builder, Dream backbone)
      dataset.py                       (PyTorch Dataset over cached shards)
      stats.py                         (Δh distribution, RMS, anisotropy)
    models/
      heads.py                         (Δh head w/ zero-init, conf head)
      variant_c.py                     (2s-sequence concat)
      variant_b.py                     (AdaLN-Zero)
      variant_a.py                     (cross-attn anchor)
    losses.py                          (MSE + KL + BCE composite)
    train.py                           (training loop, λ schedules,
                                        per-i / per-reveal-fraction logging)
    eval/
      shared_mass.py                   (reuse probe_runner metric)
      gsm8k_e2e.py                     (full Fast-dLLM swap-in eval)
    configs/
      v0_uniform_llada_C.yaml
      v0_uniform_llada_B.yaml
      v0_uniform_llada_A.yaml
      v0_uniform_dream_C.yaml          (etc.)
```

Reuse from existing tree:
- `probe_runner/hooks.py` — instrumentation pattern for capturing per-pass
  hidden states.
- `probe_runner/llada_runner.py`, `dream_runner.py` — Fast-dLLM
  forward harnesses; collect scripts will subclass / fork these.
- `T3/Think-Then-Talk/model/inference_engine.py` — for the in-block
  decode loop pattern (we'll need an analogue at inference time once
  the delta model is trained).

---

## Milestones

**M0 — characterization (no training).** Pull the cached `h_per_pass`
data we already have from `probe_runner` for ~5 GSM8K samples.
Compute the Δh = `h_i − h_1` distribution: per-layer RMS, anisotropy
(top-k singular values), correlation with reveal-fraction and `i`.
Decision gate: if Δh is too high-variance / unstructured, escalate
loss design before training. Should take <1 day of analysis.

**M1 — variant C end-to-end on a small slice.** ~1k Nemotron prompts,
LLaDA only, variant C, 2 layers. Get the loss curve down on Δh-MSE +
KL + BCE. Confirm zero-init makes step 0 = h_1-reuse baseline.
Confirm metrics-by-`i`-and-reveal-fraction logging works end-to-end.

**M2 — full data, all three variants, LLaDA.** Same loss, full training
set. Compare A / B / C on val shared-mass and on **swap-in GSM8K
accuracy** (the metric we actually care about). Report by `i` and
reveal-fraction.

**M3 — replicate winning variant on Dream-7B.** Confirms the recipe
is base-model-portable in the methodology sense (separate trained
delta models, same architecture).

Open questions deferred to post-M1 (we need data to answer):
- Whether MSE+KL+BCE jointly cover enough or we need attention-relation
  distillation (MiniLM-style) for hidden-state preservation.
- Whether 2 layers underfits — scale to 4 / 6 if M1 plateaus.
- Whether a v0.1 reveal-fraction curriculum is worth the engineering.
- Whether L2 on ‖Δh‖ is needed (only if we observe overshoot).
- Storage budget — likely need shard sub-sampling, decide once M1
  shard size is empirically known.

---

## Status

Scoping. Awaiting sign-off before code. Discussion welcome on:
storage strategy, eval set beyond GSM8K, and milestone gating
criteria (what level of GSM8K degradation is acceptable in M2).
