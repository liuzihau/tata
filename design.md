# Intra-block delta model — research idea

Brainstorm note. Informed by `intra_block_drift_findings.md` and the
`prediction_overlap` panel from `probe_runner`. Recorded here so the
direction isn't lost while the T3 axis-3 work wraps up.

> **Indexing convention in this doc.** "Pass 1" = the *first* think forward
> of a block (what T3 caches as `talk_rps`). "Pass 2, 3, …" = the
> subsequent parallel-decode forwards within the same block. In the
> probe_runner code these correspond to `pass_idx = 0, 1, 2, …`, but the
> human-friendly 1-indexed numbering is used here.

---

## Empirical anchor

LM-head shared probability mass between **pass 1** and a **later pass within
the same block**, measured per-position then averaged:

```
shared_mass(p_1, p_i) = Σ_x min( softmax(lm_head(ln_f(h_1)))(x),
                                 softmax(lm_head(ln_f(h_i)))(x) )
```

| pair | shared mass |
|------|------------:|
| pass 1 → pass 2 | ~0.80 |
| pass 1 → pass 3 | ~0.62 |
| pass 1 → pass 4 | ~0.60 |
| pass 1 → pass 5 | ~0.58 |

The pattern is consistent across **LLaDA-8B** and **Dream-7B**. Sharp drop
between pass 2 and pass 3, then near-plateau in the high-50s / low-60s.

This is the same metric used as the speculative-decoding acceptance rate in
Leviathan et al. 2023 / Chen et al. 2023 — i.e. if pass 1 were used as a
draft and pass-`i` as the target, we'd accept ~80%, ~62%, ~60%, ~58% of
positions.

---

## Why this validates that T3's talk model works at all

T3's talk model consumes `talk_rps` *cached at pass 1* and is asked to
denoise across many subsequent passes within the same block. The empirical
floor of **~58–60% LM-head overlap** between pass 1 and any later pass
means the pass-1 representation is not random — it already encodes most of
what the eventual prediction will be. The talk model only needs to recover
the residual ~40%.

That's the regime in which a small follow-up model can plausibly succeed:
the draft-vs-target overlap is high enough that the task becomes
*correction*, not *prediction-from-scratch*.

---

## Where T3's specific design leaves quality on the table

(All four critiques are about retaining the original think model's
quality. Speed/size discussion is deferred.)

1. **Fuses three layer outputs.** T3's `talk_rps` is the EAGLE3-style
   concatenation of layers `0`, `1`, and `last`. The fusion may be a good
   choice but it hasn't been carefully ablated — different layer triples
   could be better.
2. **Prunes the last two transformer layers** of the think backbone before
   feeding the talk side. The intra-block drift plots show the last few
   layers are precisely the ones that recover prediction-level
   overlap; pruning them likely costs quality the talk model has to make
   up.
3. **Iteratively updates the hidden representation** fed to the talk
   model. Each denoise step modifies `talk_rps`, which then feeds the
   next step. Errors at one step propagate into the next — classic
   accumulation problem.
4. **Generates a brand-new hidden representation** at every denoise step
   rather than predicting a *correction* to the existing one. A fresh
   prediction has to be right in absolute terms; a correction has a
   stable baseline (the think backbone's last-layer output) and only has
   to capture what changes.

Issues (1)–(3) are observations on the existing architecture. Issue (4)
is what motivates the alternative below.

---

## Proposed direction: a lightweight delta model

Replace the current talk model with one trained to predict the
**deviation** between the pass-1 final-layer hidden state and the
eventual pass-`i` final-layer hidden state. Add the deviation back to
the cached pass-1 representation; pass the result through the original
LM head.

### Inputs

- **Prefix-token KV cache** — from the original think model's pass-1
  forward (the prompt + already-decoded blocks; same KV the original
  model would attend to at pass `i`).
- **Pass-1 last-layer hidden state**, `h_1`, treated as another KV-like
  input the delta model attends to (cross-attention-style conditioning,
  not a residual stream).
- **Embedded tokens decoded during the previous iteration** — the only
  thing in `x` that's changed since the last forward, and therefore the
  signal the delta model needs to predict the new gap.

### Output

```
Δh  ∈  ℝ^{block_length × d_model}
```

A per-position correction at the **last layer's resolution** of the
think model.

### Reconstruction

```
h_i_predicted   = h_1 + Δh
logits_i        = lm_head(final_norm(h_i_predicted))
```

### Training target

Run the original think model for `i` iterations under the standard
Fast-dLLM dual-cache + parallel-decoding protocol; capture the actual
last-layer hidden state `h_i_actual`. Train the delta model to predict
`Δh = h_i_actual − h_1`. Loss: probably MSE on `Δh`, possibly with an
auxiliary KL on the LM-head logit distribution (matches the metric
that ultimately matters for accuracy).

### Inference loop (block-internal)

1. Run the original think model **once** at pass 1. Cache:
   - prefix KV (prompt + earlier blocks),
   - last-layer hidden state `h_1` for the current block.
2. For iterations 2, 3, …:
   - Embed the tokens revealed in the previous iteration.
   - Feed `(prefix KV, h_1, prev-iter token embeddings)` to the delta
     model → get `Δh`.
   - Compute `h_i_predicted = h_1 + Δh`.
   - Compute `logits_i = lm_head(final_norm(h_i_predicted))`.
   - Reveal new tokens by the standard Fast-dLLM
     confidence-thresholded transfer rule.
3. Repeat until the block is fully decoded.

The think backbone is **never re-run within a block** — only the delta
model runs at iterations 2+. That's where the speed win lives. Quality
is preserved (or at least bounded) because the prediction is anchored
to `h_1` rather than reconstructed from scratch.

---

## How this addresses the four T3 critiques

| # | T3 issue | Delta-model response |
|---|---|---|
| 1 | Fuses three layer outputs (untrained-curated layer choice) | Δh is added to **one** layer (the last). No fusion choice to ablate. |
| 2 | Prunes last two transformer layers | Uses the **full** think backbone for pass 1; no layers pruned. The delta model sits *next to* the backbone, not in place of its tail. |
| 3 | Iteratively updates the rps that feeds itself (error accumulation) | Predictions at all iterations are anchored to the **fixed** pass-1 hidden state `h_1`. The delta model never reads its own previous output, so errors cannot compound across iterations. |
| 4 | Outputs a from-scratch representation | Outputs a **correction** Δh, not a full representation. The model has a strong baseline (h_1, ~60–80% LM-head overlap with truth) and only has to learn the residual. |

---

## Open questions before scoping an implementation

1. **Is `Δh` learnable in practice?** Distribution of `h_i − h_1` needs
   to be characterised first — magnitude, anisotropy, correlation with
   simple features (last-iter revealed tokens, position-in-block, layer
   index).
2. **What input feature set is enough?** Prefix KV alone may
   underdetermine `Δh`. The previous-iteration revealed tokens are the
   obvious candidate. Whether `h_1` itself needs to be a cross-attention
   key/value bank or just a residual baseline is open.
3. **How small can the delta model be?** EAGLE / EAGLE-3 scale
   (~4–6 transformer layers, ~700M params on a 7B target) is one
   reasonable anchor.
4. **Does it transfer across base models?** A delta model trained for
   LLaDA-8B may or may not transfer to Dream / Qwen. Need a
   transferability ablation early.
5. **What's the right loss?** Pure MSE on `Δh` may be insufficient if
   small `Δh` errors get amplified by `lm_head` into large logit /
   distribution errors at high-confidence positions. Auxiliary
   distribution-level loss (KL on LM-head softmax, or `1 − shared_mass`)
   probably needed.
6. **Does it preserve the speculative-decoding-style acceptance gain?**
   I.e. when we replace the second-pass forward with `h_1 + Δh`, do we
   recover the prediction-overlap floor we measured? If yes, the delta
   model is doing its job; if no, the residual we're failing to predict
   is what costs accuracy.

---

## Status

Brainstorm. Recorded here while T3 axis 3 finishes. Discussion welcome —
this is intentionally open-ended.
