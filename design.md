# tata — intra-block delta model for masked-diffusion LMs

A research note in the style of an informal short paper. Audience: a
reader who's never seen this project and wants to understand *why* it
exists and what we're aiming for. For implementation status and code
contracts, see `engineering.md`. For how to actually run things, see
`usage.md`.

---

## 1 · Problem and motivation

Masked diffusion language models like LLaDA-8B and Dream-7B generate
text in **blocks**, with each block decoded over multiple **iterations**.
At each iteration the model runs a full forward pass over the entire
sequence, picks a few high-confidence positions to commit, and leaves
the rest masked for the next iteration. With `block_length = 32` and
up to `max_iter = 6` iterations per block, an 8-block / 256-token
generation takes up to 48 full backbone forwards. Fast-dLLM
(Niu et al. 2025; `external/Fast-dLLM/v1/llada/generate.py:132`)
amortizes this by caching the prefix KV at each block's pass-1
forward — but the in-block forwards themselves are still
backbone-sized.

**Question.** Inside a single block, how different are the per-pass
hidden states from each other? Specifically, between pass 1 (the first
forward of a block) and pass `i` (a later iteration), how much do the
model's final-layer outputs actually change?

**Answer (from the `probe_runner` measurement).** LM-head shared
probability mass — the speculative-decoding acceptance rate metric of
Leviathan et al. 2023 and Chen et al. 2023, defined as
`Σ_x min(p_pass1(x), p_passi(x))` — between pass 1 and pass `i`,
averaged per position:

| pair | shared mass |
|---|---:|
| pass 1 → pass 2 | ~0.80 |
| pass 1 → pass 3 | ~0.62 |
| pass 1 → pass 4 | ~0.60 |
| pass 1 → pass 5 | ~0.58 |

Consistent across LLaDA-8B and Dream-7B. There's a sharp drop between
pass 2 and pass 3, then a near-plateau in the high-50s / low-60s.

**The implication.** Most of what pass `i` produces is already encoded
in pass 1. If we could approximate the *delta* from pass 1 to pass `i`
with a much smaller network, we'd save N-1 full backbone forwards per
block at the cost of one small-network forward each.

---

## 2 · Idea: a lightweight delta predictor

Train a small model `f_Δ` that, given the pass-1 hidden state `h_ref`
(the most recent backbone forward in this block) and the current
revealed-token pattern, predicts the **delta** `Δh` such that
`h_ref + Δh ≈ h_target`, where `h_target` is what the backbone *would*
have produced at some later iteration `i_target`. Reconstruct the
final-layer hidden via `h_predicted = h_ref + Δh`, pass it through the
frozen `lm_head(final_norm(·))`, and use the resulting logits to commit
tokens via the standard Fast-dLLM transfer rule.

At inference, the backbone runs once at the start of each block (pass 0)
and refreshes h_ref via a partial-forward rollback only when a
confidence head says the prediction is suspect. The delta model handles
all other in-block iterations.

### Why this might work

- **Anchored prediction beats from-scratch prediction.** The delta model
  doesn't have to reconstruct the residual stream; it only has to
  correct a baseline that's already 60-80% right (per shared-mass
  numbers above). Strong inductive prior.
- **No error accumulation across iterations.** Every iteration's
  prediction is computed from the *same* h_ref captured at pass 1 — the
  delta model never reads its own output. Compare against approaches
  that iteratively update a draft representation (e.g. T3's talk-model
  loop), which suffer from compounding drift.
- **Zero-init friendly.** With the Δh projection initialized to zero
  (ControlNet / Flamingo recipe), the model at step 0 emits `Δh = 0`,
  i.e. `h_predicted = h_ref`. That's exactly the "reuse pass-1 output
  for all later passes" baseline. Training can only do better.

### Inputs to `f_Δ`

| input | provenance |
|---|---|
| `h_ref` ∈ ℝ^{32 × d_model} | backbone's last-layer hidden state at the block, from pass 1 (or a rollback) |
| `prev_emb` ∈ ℝ^{32 × d_model} | per-position token embedding of the *current* x state — revealed tokens use their actual ID, unrevealed positions use the mask-token embedding |
| `prefix_kv` ∈ ℝ^{2 × n_kv_heads × W × d_head} | cached K/V at the last layer for the W tokens just before the block (W = 32 at train, up to 64 stored) |
| `block_start_pos` ∈ ℤ | absolute position of the block's first token in the sequence; used to drive RoPE at training-aligned positions |

### Outputs

- `Δh` ∈ ℝ^{32 × d_model} — per-position correction added to `h_ref`.
- `c_pos` ∈ (0,1)^{32} — per-position confidence. At inference, a
  position commits only if Fast-dLLM's transfer rule selects it AND
  `c_pos[i] ≥ per_pos_threshold` ("agreement decoding"; §4 below).

### Reconstruction

```
h_predicted = h_ref + Δh
logits      = lm_head(final_norm(h_predicted))
```

### Training target

Run the backbone for up to `max_iter = 6` iterations under Fast-dLLM's
prefix-cache decoding protocol; cache `h_per_pass[1..6]` and the
per-iter reveal patterns. For each `(i_ref, i_target)` pair with
`1 ≤ i_ref < i_target ≤ 6`, supervise:

```
L = λ_mse · MSE(Δh_pred, h_target − h_ref)
  + λ_kl  · KL(p_actual ‖ p_predicted) at mask positions
  + λ_conf · BCE(c_pos, shared_mass(p_actual, p_predicted)) at mask positions
```

where `p_actual = softmax(lm_head(final_norm(h_target)))` and similarly
for `p_predicted`. MSE preserves hidden-state magnitude; KL preserves
the distribution that ultimately matters (small Δh errors at
high-confidence positions can blow up into large logit errors); the
confidence-head loss trains a per-position gate for inference.

---

## 3 · Locked design (M1)

| axis | choice | rationale |
|---|---|---|
| Per-base-model | LLaDA-8B and Dream-7B trained separately | No cross-model transfer goal yet. |
| Block size | 32 (Fast-dLLM default) | Matches the baseline. |
| Δh resolution | every block position (32) | CC (already-committed) positions learn Δh ≈ 0 organically. |
| Conditioning layer | last layer only | M1 simplification. Multi-layer fusion (EAGLE-3) deferred to M2. |
| Prefix-KV scope | last layer, last `W = 32` tokens (stored up to 64) | Limits cache size; compensates with strong `h_ref` (carries earlier context via bidirectional attention). |
| Δh head init | zero-init | At step 0, model = "reuse h_ref" baseline. |
| Model size | 2 transformer layers, d_model matched to backbone | Scale up only if M1 plateaus. |
| Iteration-index embedding | **none** | `Δh` is deterministic given `h_ref` + reveal pattern. |
| Decoding mode | `factor = 1.0` (Fast-dLLM dynamic per-rank threshold) | Paper-recommended; must match between collect and eval. |
| Sampling at training | uniform over `(i_ref, i_target)` pairs | No curriculum unless v0 plateaus. |

Loss weights at M1: `λ_mse = 0.1`, `λ_kl = 1.0`, `λ_conf = 1.0`. (See
§2.4 of `engineering.md` for the back-of-envelope that motivates this
balance.)

---

## 4 · Architecture variants

All three share: 2 transformer layers, d_model = backbone d_model,
n_heads matched to backbone, zero-init Δh head, per-position confidence
head. They differ in how `h_ref` and `prev_emb` are combined.

| variant | mechanism | rationale |
|---|---|---|
| **C** (M1 default) | concat `h_ref ⊕ prev_emb` along the sequence axis (length 2·32); segment embed distinguishes the two halves; self-attn + cross-attn into prefix_kv at each layer; take the prev_emb half for the Δh head | Simplest. If A/B don't beat this, the conditioning style isn't doing work. |
| **B** (M2 candidate) | `prev_emb` is the residual stream; `h_ref` pooled to predict per-layer AdaLN-Zero `(γ, β)` | Param-efficient. Lit hint: AdaLN-Zero wins at low-dim conditioning. |
| **A** (M2 candidate) | `prev_emb` as residual stream; `h_ref` as frozen K/V bank for gated cross-attention (Flamingo-style) | Lit's default for high-dim conditioning. Closest match to "h_ref is the anchor; prev_emb is the perturbation." |

Implementation order: C → B → A (cheapest debug → richest model).

---

## 5 · Alignment chain (collect ↔ baseline ↔ inference)

This is load-bearing: the delta model's training-target distribution
must equal what the inference baseline produces, or comparisons lose
meaning.

| stage | pass 0 of block | iter ≥ 1 within block | "redo" within block |
|---|---|---|---|
| baseline `generate_with_prefix_cache` | full forward | partial forward w/ cached prefix | n/a |
| collect (`collect_llada.py`) | full forward + prefix_kv extraction + cache_to_s | partial forward w/ cache_to_s, capture h_per_pass[i] | n/a |
| inference (`hybrid_runner.py`) | full forward, capture h_ref + prefix_kv + cache_to_s | delta forward (no backbone) | partial forward w/ pass-0 cache (rollback) |

All three paths use the same `model(x[:, s:], past_key_values=past_to_s, use_cache=True)` call whenever they need a backbone hidden state for an in-block update. h_per_pass[i] (training target), baseline h, and rollback-refreshed h_ref are all from the same distribution.

This alignment surfaced two real bugs and one false alarm during
development; the full postmortem lives in `engineering.md` §6.

---

## 6 · Milestones

- **M0 — characterization (done, lives in `probe_runner`).** Measure
  Δh = h_i − h_1 distribution from cached probe data. Decision gate:
  if Δh is too high-variance or unstructured, redesign loss before
  spending compute. Result: distribution is structured; proceed.
- **M1 — variant C end-to-end, LLaDA-8B, 5000 prompts.** Loss curve
  drops; zero-init invariant holds at step 0; mid-train GSM8K stays
  within ε of vanilla `generate_with_prefix_cache`.
  - Trial 1: GSM8K 0.72 baseline → 0.10 at 15k (post-mortem in §3.1
    / §3.2 of `engineering.md`).
  - Trial 2 (after BCE fix, agreement decoding, alignment audit):
    GSM8K 0.28 → **0.44** at later checkpoint. Real progress; still
    0.28 short of baseline 0.72.
- **M1.5 — Tier 1 fixes (loss-space + sampler + dropout). Landed
  2026-05-12. Result: neutral.** Three changes shipped together
  to test the loss-space hypothesis:
  - **T1** — MSE in `final_norm` space (with `λ_mse 0.1 → 1.0`).
  - **T2** — i_ref-biased `WeightedRandomSampler` (multiplier 3 →
    60% P(i_ref=0)).
  - **T3** — dropout 0.1 in the delta model.

  Final metrics: train MSE/KL/BCE 0.989/0.092/0.348, val 1.081/0.293/0.472.
  Compared to trial-2: **val KL identical (0.295 → 0.293)**; train-val
  KL gap narrowed 3.93× → 3.19× (T3 working but undersized); MSE
  scale dropped from O(10²) to O(1) as predicted (T1 mechanically
  correct). GSM8K **0.40** vs trial-2's 0.44 — within 1σ noise at
  n=200 (SE ≈ 0.035).

  **Conclusion: loss-space was not the binding constraint at this
  data scale.** Tier-1 was mechanically successful (clean loss, in
  the right space, dropout reduced gap) but did not move the needle
  on Nemotron generalization. The lever is data-vs-capacity ratio,
  not loss space.

  Why this is consistent: overfitting onset was at ~1.5k steps —
  1500 × 256 = 384k samples ÷ 40k unique blocks = **~9.6 block-views**.
  The 15 (i_ref, i_target) pairs in a block share `h_per_pass`,
  `prefix_kv`, `reveal_per_pass`, and the prompt's `substituted_ids`
  template, so the **block** is the effective sampling unit, not the
  pair. At ~10 block-epochs the network can start memorizing
  block-specific structure with 500M params. By step 20k that's ~128
  block-epochs — too many for the data budget.

- **M1.6 — `λ_mse = 0` ablation.** Test whether MSE is doing any
  load-bearing work at all. M1.5 val KL was identical to trial-2's
  despite swapping MSE-space, so MSE's contribution to generalization
  is suspect. Config-only run on the existing cache; can launch in
  parallel with M2 collect.

- **M2 — Tier 2 (data scaling + capacity + variants).** Now ranked
  with T5 as primary:
  - **T5 — scale data (5000 → 15000–20000 prompts). PRIMARY.**
    Direct response to the block-as-sampling-unit overfitting
    signature. Pure collect cost (no code change). Lands in a new
    cache dir (`cache_v1_20k/llada/`) so it doesn't race with M1.6
    on the original cache. Hash-stable test split means the 800
    test prompts are identical to trial-2's.
  - **T7 — deeper delta model (2 → 4 layers).** Only if T5 plateaus
    below 0.60; capacity may need to grow alongside data.
  - **Variants A (cross-attn anchor) / B (AdaLN-Zero).** Compare on
    val-shared-mass and swap-in GSM8K accuracy, binned by gap and
    reveal-fraction.
  - **T8 (= old C) — sliced 1-D distribution-matching regularizer**
    on `h_pred` vs `h_target`. SIGReg's projection-based mechanism,
    retargeted from anti-collapse to anti-drift. Gate: only attack
    if per-bin val diagnostic shows late-iter / high-reveal
    degradation after T5 lands.

- **M3 — replicate winning variant on Dream-7B.** Confirms the recipe
  is base-model-portable in methodology (separate trained delta
  models, same architecture).

### Out-of-tier eval action

**T4 — `per_pos_threshold` sweep at the current checkpoint.** The
0.85 default at `gsm8k_per_pos_threshold` was picked at design time,
not tuned. Hybrid GSM8K accuracy is non-monotone in threshold (low =
bad commits, high = unnecessary rollbacks). Run the existing eval
harness's sweep over {0.70, 0.80, 0.85, 0.90, 0.95} on the M1.5
checkpoint. Free, ~30 min. Could recover the trial-2/M1.5 GSM8K gap
at zero training cost. Command in `usage.md`.

---

## 7 · Open research questions

1. **Is Δh learnable in practice?** M0 said yes, but the eventual M1
   GSM8K curve is the real answer.
2. **What feature set suffices?** Prefix KV alone might
   underdetermine Δh. `prev_emb` is the obvious extra signal. Whether
   `h_ref` itself wants a cross-attention bank or just a residual
   baseline is the A-vs-C question.
3. **How small can the delta model be?** EAGLE / EAGLE-3 scale
   (4–6 transformer layers, ~700M params on a 7B target) is one
   reasonable anchor. M1 starts at 2 layers.
4. **Does it transfer across base models?** A delta trained for
   LLaDA-8B may or may not transfer to Dream-7B. M3 tests this — we
   expect *no* transfer (architectures differ in subtle ways), but the
   methodology should transfer.
5. **Right loss balance?** Pure MSE on Δh is insufficient when small
   Δh errors amplify into large logit errors at high-confidence
   positions. The KL term addresses this. Whether the per-position
   BCE confidence loss helps in practice is the M1 trial-2 question.
6. **Does it preserve the speculative-decoding acceptance gain?** If
   we replace pass `i`'s backbone with `h_1 + Δh`, do we recover the
   measured shared-mass floor (~58–80%)? If yes, the delta model is
   doing its job. If no, the residual we're failing to predict is
   what costs accuracy.
7. **What is the right loss space?** MSE in `losses.py` operates on
   raw h (every coordinate weighted equally), but `lm_head` reads via
   RMSNorm + a linear that weights directions very unevenly — some
   h-coordinates contribute nothing to the logits, so MSE gradient on
   them is wasted. Is post-`final_norm` MSE strictly better? Is MSE
   necessary at all once KL is in place? M1.5 (T1) and M1.6 (B) test
   these. LeWM (arXiv:2603.19312) makes the analogous point on its
   end: the LN'd CLS embedding is the wrong space for their
   anti-collapse regularizer, fixed by a learned post-LN projector.
   Trial-2 evidence supports T1 directly: train-val gap is ~4× on KL
   (post-`lm_head` space) but only ~1.2× on MSE (raw-h space), i.e.
   MSE has saturated at the fp16 noise floor while KL is still
   learning generalizable structure.
8. **Train-inference distribution mismatch in `i_ref`.** The dataset
   enumerates all `(i_ref, i_target)` pairs with `i_ref < i_target`;
   only 5/15 = 33% have `i_ref = 0`. At inference, `h_ref` is captured
   at pass 0 and refreshed only via rollback, so the inference
   distribution is ~80–95% `i_ref = 0` (rate-of-rollback-dependent).
   The model spends ⅔ of its gradient on pairs it almost never sees
   at inference. M1.5 (T2) addresses this with a weighted sampler.
   Open: what's the *actual* inference `i_ref` distribution per
   rollback rate? Diagnostic to land in M1.5 trial.

---

## 8 · How this relates to T3 and prior work

Closest prior work: **EAGLE-1** (arXiv:2401.15077) — predicts
hidden-state corrections for speculative decoding on autoregressive
LMs. We take the same shape of idea (lightweight head predicts a hidden
correction; reconstruct logits via the frozen LM head) but apply it to
the diffusion-decoding regime, where the structure is "many iterations
of the same block" rather than "many tokens of an autoregressive
sequence."

Design hints from elsewhere:
- **ControlNet / Flamingo** — zero-init the output projection so the
  base behavior is preserved at step 0.
- **DiT (AdaLN-Zero)** — variant B's modulation recipe.
- **EAGLE-3** — multi-layer hidden fusion. Deferred to M2.
- **LeWM (arXiv:2603.19312, Maes et al. 2026).** A JEPA world model
  whose two-term loss (next-embedding prediction + an anti-collapse
  SIGReg term) yielded three transferable lessons:
  **(A)** compute the regression loss in a space the downstream
  consumer actually reads (their LN-escape projector → our
  `final_norm`-aligned MSE, M1.5);
  **(B)** fewer well-chosen loss terms beat many heuristics (their
  2-term recipe → our `λ_mse = 0` ablation, M1.5);
  **(C)** SIGReg's sliced-projection mechanism (random unit
  directions + a 1-D distribution-matching statistic + Cramér–Wold)
  is reusable beyond Gaussian targets — we retarget it from
  anti-collapse to anti-drift, matching `h_pred`'s batch-marginal
  distribution to `h_target`'s (M2).
  The bulk of LeWM's machinery (SIGReg's N(0, I) target, end-to-end
  encoder training, no-stop-grad / no-EMA stability tricks) doesn't
  port because we are supervised with a frozen encoder — collapse is
  not our failure mode. Only the loss-space lessons and the
  random-projection mechanism transfer.

T3 (this repo's earlier "Think-Then-Talk" line) has a "talk model" that
denoises an EAGLE-3-style fused hidden representation. The four design
critiques that motivated the delta-model proposal:

1. T3 fuses three layer outputs without ablating which triple is best.
   The delta model uses one layer (last) — no fusion choice to defend.
2. T3 prunes the last two transformer layers of the think backbone.
   Empirically those layers are where prediction-overlap recovers.
   The delta model attaches *beside* the full backbone, prunes nothing.
3. T3 iteratively updates the representation that feeds itself,
   accumulating errors. The delta model is anchored to a fixed h_ref
   per block — errors cannot compound across iterations.
4. T3 outputs a from-scratch representation. The delta model outputs
   a *correction*, leveraging a strong baseline.

---

## 9 · Status snapshot (2026-05-12)

M0 done. M1 trial 1 reproduced an alarming GSM8K decline
(0.72 baseline → 0.10 at step 15k). M1 trial 2: GSM8K **0.28 → 0.44**
after BCE clamp + agreement decoding + alignment audit. M1.5 Tier-1
(T1+T2+T3): val KL identical to trial-2 (0.293 vs 0.295), GSM8K
**0.40** (within 1σ noise of 0.44).

**The diagnosis updated.** The loss-space hypothesis was *good
reasoning from trial-2 evidence* but is not the binding constraint
at this data scale. Tier-1's mechanically correct loss reallocation
did not move val KL, so the bottleneck is upstream — in data
volume relative to model capacity. Concretely: overfitting onset at
~1.5k steps corresponds to ~9.6 block-views (384k samples / 40k
unique blocks), and the 15 pairs per block share enough structure
(h_per_pass stack, prefix_kv, reveal patterns, substituted_ids
template) that the **block** is the effective sampling unit. At
~10 block-epochs a 500M-param network starts memorizing
block-specific structure faster than it generalizes.

Three concurrent next actions:

1. **T4 sweep on the M1.5 checkpoint** — free, ~30 min, may recover
   the trial-2 / M1.5 GSM8K delta entirely from a better operating
   point.
2. **M1.6 = T6 (`λ_mse = 0` ablation)** — config-only, runs against
   the *existing* `cache_v1/llada/` cache. Tests whether MSE is
   load-bearing at all.
3. **M2 collect for T5** — collect 15000–20000 prompts into a new
   `cache_v1_20k/llada/` directory. Can run in parallel with M1.6
   without racing on cache files. The hash-stable test split keeps
   the same 800 test prompts as trial-2 for direct comparison.

See `usage.md` Tier-1+ recipe for commands.
