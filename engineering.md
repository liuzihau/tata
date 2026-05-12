# tata — engineering reference

For an agent or developer who is going to read the code and continue
the work. Organized by *surface area* (data / model / loss / training /
inference / eval / sanity tests), with each section laying out (a) the
current contract, (b) recent decisions, and (c) open items.

For *why* any of this exists, see `design.md`.
For *how* to run anything, see `usage.md`.

---

## Status snapshot (2026-05-12)

Current focus: **two parallel runs + one free eval** —
- T4 sweep on the M1.5 checkpoint (free, ~30 min).
- M1.6 = T6 (`λ_mse = 0`) on the existing `cache_v1/llada/` cache.
- M2 T5 collect (5000 → 20000 prompts) into a separate
  `cache_v1_20k/llada/` directory to avoid racing M1.6 on the
  manifest.

| area | state |
|---|---|
| Data schema | v2 (`prefix_kv` field, `prefix_window=64` stored, optional `prefix_kv_pad_mask`). Trial-2 / M1.5 cache reused for M1.6. T5 collect lands in a new dir. |
| Model | VariantC with RoPE on absolute positions (§1.5), RMSNorm + SwiGLU primitives (§1.6), per-position confidence head, dropout knob exposed (T3, §2.5). |
| Training | Preload-all-samples (~80 GiB RAM). Single-process loader. Optional i_ref-biased `WeightedRandomSampler` (§3.4, T2). |
| Loss | Composite MSE + KL + BCE. MSE space configurable via `loss.mse_space` (`raw` legacy, `final_norm` recommended; §3.2). BCE target clamped to [0, 1]. |
| Inference | Hybrid runner: full forward @ pass 0, delta @ iter ≥ 1, partial-forward rollback (matches collect's iter ≥ 1). Agreement decoding via per-position threshold. |
| Eval | gsm8k_e2e harness sweeps `per_pos_thresholds`; reports hybrid/vanilla accuracy + speedup + disagreement count. |

Trial results history:
- **Trial 1** (pre-3.1 rollback fix, block-aggregate conf): 0.72 baseline → 0.20 at 15k.
- **Trial 2** (post-3.1 alignment audit, agreement decoding, BCE clamp): GSM8K 0.28 → **0.44**. Train MSE/KL/BCE 12.6/0.054/0.29 vs val 16.1/0.38/0.59 — train-val KL gap ~4× hypothesized to be loss-space misallocation.
- **M1.5 Tier-1** (T1 final_norm MSE + T2 sampler + T3 dropout): final train MSE/KL/BCE 0.989/0.092/0.348 vs val 1.081/0.293/0.472. **Val KL identical to trial-2 (0.293 vs 0.295)**; KL gap narrowed 3.93× → 3.19× (T3 working, undersized); GSM8K **0.40** (within 1σ of 0.44). Loss-space hypothesis **rejected as primary lever** at this data scale. Overfitting onset at ~1.5k steps = ~9.6 block-views (384k samples / 40k unique blocks); the 15 pairs/block share enough structure that the *block* is the effective sampling unit, putting a 500M-param model at ~10 block-epochs of effective data by 1.5k.

Recent changes log (most recent first):

- **2026-05-12** M1.5 Tier-1 result recorded; tier ranking updated. T5 (data scaling) promoted to PRIMARY for M2; T6 (`λ_mse = 0`) elevated to runnable parallel ablation (M1.6).
- **2026-05-12** §3.2 T1 (= M1.5 A) landed: `mse_space="final_norm"` option in `composite_loss`; new config (`m1_5_llada_variant_c.yaml`) rescaled `λ_mse 0.1 → 1.0`.
- **2026-05-12** §3.4 T2 landed: i_ref-biased `WeightedRandomSampler` (`data.ref0_weight_multiplier`, default 3.0).
- **2026-05-12** §2.5 T3 landed: `model.dropout = 0.1` in M1.5 config (was 0.0 in trial-2).
- **2026-05-11** Fix BCE NaN crash: clamp `c_label_per_pos ∈ [0, 1]` in `losses.py` (fp32-softmax roundoff).
- **2026-05-11** §3.2 per-position confidence + agreement decoding landed.
- **2026-05-11** §3.5 pad-and-mask short prompts (no skipping).
- **2026-05-11** §3.4 `COLLECT_PREFIX_WINDOW=64` (training reads last 32).
- **2026-05-11** §3.1 alignment chain audited — original collect was correct; hybrid runner rollback fix (full → partial) is the real change.
- **2026-05-11** §1.5+§1.6 variantc rewrite (RoPE / RMSNorm / SwiGLU).
- **2026-05-10** §1.1–§1.7 data-loading rewrite; preload-into-RAM kills I/O bottleneck.
- **2026-05-09** Wallclock timers in train loop.

---

## File map

```
tata/
  design.md                                — theory / motivation / research framing
  engineering.md                            — this file
  usage.md                                  — operator's manual
  delta_model/                              — the importable package
    llada_runtime.py                        — load_llada + Fast-dLLM helpers
    data/
      schema.py                             — cache-format constants
      collect_llada.py                      — cache builder
      dataset.py                            — TataDeltaDataset (preload+shard+per-sample)
      repack.py                             — convert per-sample files to multi-sample shards
      sample_prompts.txt                    — built-in smoke-test prompts
    models/
      heads.py                              — DeltaHead, ConfHead (legacy), ConfHeadPerPos
      variant_c.py                          — VariantC (RoPE, RMSNorm, SwiGLU, per-pos conf)
    losses.py                               — composite_loss (MSE + KL + per-position BCE)
    train.py                                — AdamW + cosine + wandb + timers
    inference/hybrid_runner.py              — generate_with_delta + agreement decoding
    eval/
      shared_mass.py                        — overlap metric
      gsm8k_e2e.py                          — end-to-end eval harness
    sanity/
      test_zero_init.py                              — T2: model+loss invariants
      test_collect_roundtrip.py                      — T3: cache schema check
      test_partial_full_forward_equivalence.py       — T4: documents Fast-dLLM partial-forward behavior
      inspect_llada_modeling.py                      — diagnostic for the loaded LLaDA modeling code
    configs/
      m1_llada_variant_c.yaml               — M1 trial-2 baseline (raw MSE, dropout 0, uniform sampler)
      m1_5_llada_variant_c.yaml             — M1.5 Tier-1 (T1+T2+T3) — final_norm MSE, dropout 0.1, ref0-biased sampler
      m1_6_llada_variant_c.yaml             — M1.6 = T6 (λ_mse = 0 ablation) — KL+BCE only loss; rest matches M1.5
```

---

## 1 · Data pipeline

### 1.1 Cache schema (`data/schema.py`)

`SCHEMA_VERSION = 2`. Per-sample file is a `torch.load`-able dict:

```
{
  "prompt_token_ids":      [Lp]              long
  "generated_token_ids":   [GEN_LENGTH=256]  long
  "prompt_len":            int
  "blocks":                list[dict] of NUM_BLOCKS=8 entries
  "meta":                  dict
  "record":                dict
}
```

Each `blocks[b]` entry:

```
{
  "prefix_kv":           [2, n_kv_heads, prefix_window, d_head]   fp16
                          ^ K and V stacked; tokens span [s-prefix_window, s)
                            at block start s. Front-padded with zeros if
                            prompt_len < prefix_window.
  "prefix_kv_pad_mask":  [prefix_window]  bool   True = real slot, False = padded
  "h_per_pass":          [MAX_ITER=6, BLOCK_LENGTH=32, d_model]   fp16
                          last-layer hidden at the block region, per iter
  "reveal_per_pass":     [MAX_ITER=6, BLOCK_LENGTH=32]            bool
                          reveal state at the START of each pass
  "n_passes_actual":     int    actual iters used (≤ MAX_ITER); slots beyond are padded
}
```

`meta` carries `prefix_window`, `schema_version`, decoding mode, d_model,
n_kv_heads, etc. The shard manifest (`shards_manifest.json`) and the
per-cache manifest (`manifest.json`) record `prefix_window` so `T3`
(`test_collect_roundtrip`) validates shapes against the actual stored
size, not a hardcoded constant.

Two window constants: `PREFIX_WINDOW = 32` (what training/inference
consume) and `COLLECT_PREFIX_WINDOW = 64` (what collect stores). Dataset
slices `[..., -PREFIX_WINDOW:, :]` before returning so any cache with
stored window ≥ 32 is loadable at training time.

### 1.2 Collector (`data/collect_llada.py`)

Forks Fast-dLLM v1's `generate_with_prefix_cache`. Per prompt:

```
for nb in range(NUM_BLOCKS=8):
    s = prompt_len + nb * 32; e = s + 32
    past_key_values = None
    i = 0
    while True:
        if i == 0:
            full forward `model(x, use_cache=True)`
              → capture h_per_pass[0] via hook (slice [s:e])
              → past_key_values from out; slice to [:s]
              → prefix_kv  ← past_key_values[-1][..., s-W:s, :]  (front-pad if s<W)
              → prefix_kv_pad_mask
              → token transfer on full-seq logits, restricted to [s:e]
        else:
            partial forward `model(x[:, s:], past_key_values=past_key_values, use_cache=True)`
              → capture h_per_pass[i] via hook (slice [:BL])
              → token transfer on suffix logits, restricted to block
        i += 1
```

CLI: `--prefix_window` (default 64), `--factor` / `--threshold` for
decoding mode (mutually exclusive; default `factor=1.0`).

### 1.3 Dataset (`data/dataset.py`)

Auto-detects storage layout at construction:

- If `cache_root/shards_manifest.json` exists → **shard mode**. LRU caches *shards* (~800 MiB each).
- Else → **per-sample mode**. LRU caches single sample files (~16 MiB each).

`preload=True` (default in M1 config) defeats the random-shuffle locality
miss: at `__init__` every sample/shard that contributes any index is
pinned in RAM without eviction. ~80 GiB at 5000 train samples. Drops
samples whose pairs are filtered out by `train_filter` / `val_filter`.

Per `__getitem__` returns:

```
{
  "h_ref":               [BL, d_model]            fp16
  "h_target":            [BL, d_model]            fp16
  "prefix_kv":           [2, n_kv_heads, BL, d_head]   fp16
  "prefix_kv_pad_mask":  [BL]                     bool
  "substituted_ids":     [BL]                     long   (mask_id at non-revealed; real ID at revealed)
  "mask_tgt":            [BL]                     bool
  "block_start_pos":     int                              = prompt_len + b * BL
  "i_ref":               int
  "i_target":            int
  "reveal_frac":         float
}
```

Notable: `prev_emb` is *not* returned — it's computed on GPU after
batching via `token_embed(substituted_ids)` in `train.py` / `hybrid_runner.py`.
This saved 128 MiB of CPU→GPU transfer per batch (§1.2 of the
improvement log).

### 1.4 Open items in data

- **§2.1 shm raise** (blocked on cluster admin) — would unlock
  `num_workers ≥ 2` and `pin_memory=True`. After preload landed this is
  a small additional speedup, not load-bearing.
- **Locality-aware sampler** (proposed, not done) — would let us roll
  back from preload-everything to per-shard caching if cache grows
  past RAM. Custom `Sampler` that bursts indices grouped by source
  sample/shard.

---

## 2 · Model (`models/variant_c.py`)

### 2.1 Architecture (post §1.5 + §1.6)

```
Inputs:
  h_ref               [B, 32, d_model]                        fp16/bf16
  prev_emb            [B, 32, d_model]                        (built on GPU)
  prefix_kv           [B, 2, n_kv_heads, 32, d_head]          fp16
  block_start_pos     [B]                                     long
  prefix_kv_pad_mask  [B, 32]                                 bool

Outputs:
  delta_h             [B, 32, d_model]
  c_pos               [B, 32]   ∈ (0, 1)
```

Block structure (`_DeltaBlock`), 2 layers by default:
1. pre-RMSNorm → self-attn over `[h_ref ; prev_emb]` (length 64) with RoPE
2. pre-RMSNorm → cross-attn (Q from variantc body, K/V from prefix_kv passed through), Q rotated by RoPE; key-padding mask from `prefix_kv_pad_mask`
3. pre-RMSNorm → SwiGLU FFN
Final RMSNorm before heads.

### 2.2 RoPE details (§1.5)

- Positions are *absolute*: row `i` of both `h_ref` and `prev_emb` halves rotate at `block_start_pos + i`. Cross-attn queries rotate at the same positions; prefix_kv K is passed through *as-is* (already RoPE-rotated by the backbone at `[s−32, s)`; double-rotation would corrupt it).
- `rope_theta` is read from `model.config.rope_theta` at startup (1e6 default for LLaDA-8B), not hardcoded. `rms_eps` is read from `model.config.layer_norm_eps`. Both surface as `bb_cfg` in `train.py`.
- RoPE module has no buffers — `inv_freq` is recomputed on the fly per call in fp32, sin/cos computed in fp32, cast to `x.dtype` only at the final multiply. Avoids the bf16-downcast issue that `register_buffer` introduces.
- The "concat-half" rotate convention matches LLaDA's `RotaryEmbedding.rotate_half`.

### 2.3 Primitives (§1.6)

- `_RMSNorm` matches LLaDA's eps; computed in fp32 internally with cast back.
- `_SwiGLU` FFN with `d_ff_inner ≈ (8/3) · d_model ≈ 10944` (rounded to multiple of 32) — keeps total FFN params close to the previous GELU 4·d_model budget.

### 2.4 Heads (`models/heads.py`)

- `DeltaHead` — `Linear(d_model, d_model)` with **zero-init weight and bias**. At step 0, emits Δh ≡ 0 → `h_predicted ≡ h_ref`. Load-bearing invariant: T2 (`test_zero_init`) asserts `delta_h.abs().max() == 0`.
- `ConfHead` (legacy, kept for back-compat) — pooled features → scalar.
- `ConfHeadPerPos` (§3.2) — per-position features → `[B, BL]` sigmoid. Same MLP shape as `ConfHead` but no pooling. Current default.

### 2.5 Open items in model

- **Variants A and B** (M2) — cross-attn anchor and AdaLN-Zero. Same input/output signature; only inner mixing differs.
- **Multi-layer fusion** (M2) — EAGLE-3-style. Requires recollect with multi-layer hidden states.

---

## 3 · Loss (`losses.py`)

`composite_loss` returns dict with `loss` (grad-bearing) plus detached
component scalars for logging.

```
L = λ_mse · MSE(Δh_pred, h_target − h_ref)
  + λ_kl  · KL(p_actual ‖ p_predicted)        at mask positions only
  + λ_conf · BCE(c_pos, shared_mass)          at mask positions only
```

- `p_actual = softmax(lm_head(final_norm(h_target)))` (detached).
- `p_predicted = softmax(lm_head(final_norm(h_ref + Δh_pred)))`.
- `shared_mass[i] = Σ_v min(p_actual[i, v], p_predicted[i, v])`. Per-position; detached for the BCE target.
- BCE computed in fp32 for stability (`c_pos.float()` + label.float()).
- Aggregate metrics (`c_pred_mean`, `c_label_mean`) derived from per-position tensors averaged over mask positions — same scale as the old block-aggregate scalars, comparable in wandb across changes.

### 3.1 Loss weights at M1

Current: `λ_mse = 0.1`, `λ_kl = 1.0`, `λ_conf = 1.0`. Picked after observing
raw scales at the start of training (MSE ~20, KL ~0.45, BCE ~0.7).
Goal was to put all three components on the same order of magnitude
so each contributes meaningful gradient signal.

### 3.2 MSE space — `raw` vs `final_norm` (T1; landed 2026-05-12)

Motivation: `design.md` §6 M1.5 T1 + §7 Q7 + §8 LeWM bullet, plus
trial-2 evidence (train-val KL gap ~4× while train-val MSE gap is
only ~1.2× — MSE has saturated at the fp16 noise floor in raw-h
space and is consuming 76% of weighted gradient without driving the
quantity the lm_head actually reads).

`composite_loss(..., mse_space=...)` switches between:

- **`mse_space="raw"`** (legacy): `MSE(Δh_pred, h_target − h_ref)`.
  This is the trial-1/2 behavior. Default kwarg of `composite_loss`
  for API back-compat (so T2/test_zero_init keeps passing).
- **`mse_space="final_norm"`** (T1, M1 config default):
  `MSE(final_norm(h_ref + Δh_pred), final_norm(h_target))`.
  Aligns MSE gradient with the subspace `lm_head` reads.
  `final_norm(h_target)` runs under `torch.no_grad()` and reuses the
  same forward already needed for the KL path, so the only added
  cost is one RMSNorm on `h_pred` (negligible).

Scale rescale on T1: raw-h MSE was O(10²), post-`final_norm` MSE
is O(1) — `m1_5_llada_variant_c.yaml` shifts `λ_mse 0.1 → 1.0` so the
MSE term contributes a comparable fraction of the total loss as KL.
Re-tune after the first stable run; if MSE term ends up < 1% or
> 50% of the total, retune λ_mse.

Zero-init invariant under T1: `delta_h ≡ 0` at step 0 ⇒ `h_pred =
h_ref` ⇒ `MSE(final_norm(h_ref), final_norm(h_target))` — non-zero
but finite, same shape as before (`test_zero_init` checks raw-h
MSE specifically so leaving its `composite_loss(...)` call without
`mse_space=` keeps the existing assertion valid).

### 3.3 Sliced distribution-matching regularizer (M2 candidate, T8 = old C)

Motivation: `design.md` §6 M2, §8 LeWM bullet (C). Trajectory drift
is an accepted M1 limitation (§9 below) — once delta commits a wrong
token, subsequent `prev_emb` and `h_pred` diverge from any backbone
trajectory. Idea: regularize `h_pred`'s **batch-marginal
distribution** to match `h_target`'s, using SIGReg's
random-projection mechanism but with the empirical `h_target`
distribution as target instead of N(0, I).

Sketch:

```python
# h_pred, h_target both [B, T, d_model]
flat_pred   = (h_ref + delta_h_pred).reshape(-1, d_model)        # [B*T, d]
flat_target = h_target.detach().reshape(-1, d_model)             # [B*T, d]
u           = torch.randn(M, d_model, device=...)
u           = u / u.norm(dim=-1, keepdim=True)                   # [M, d]
proj_pred   = flat_pred   @ u.t()                                # [B*T, M]
proj_target = flat_target @ u.t()                                # [B*T, M]
# 1-D Wasserstein-1 per projection: sort + mean |diff|
sw1         = (proj_pred.sort(dim=0).values
               - proj_target.sort(dim=0).values).abs().mean()
L          += λ_drift * sw1
```

This is sliced Wasserstein-1, not Epps–Pulley — Wasserstein-1 is
simpler (no quadrature) and matches a *non-Gaussian* target, which is
what we need. Cramér-Wold gives the same theoretical guarantee:
matching all 1-D marginals ⇒ matching the joint, asymptotically in
M.

Open knobs:
- `M` (number of projections): 256 / 1024 / 4096. LeWM finds
  insensitivity above ~256; start at 1024.
- `λ_drift`: 0.01 / 0.1 / 1.0. Likely small — this is auxiliary.
- Where to attach: only `h_pred` from the predicted side, but
  potentially also on the delta model's internal residual stream if
  we observe degenerate internal activations.

Cost: M random projections per step, two sorts of `[B*T, M]`
tensors. With B=256, T=32, M=1024: 8192 × 1024 = 8.4M floats sorted
per step (cheap on GPU).

Gate to enabling: only meaningful if Tier 1 (T1+T2+T3) shows that
the current eval bottleneck is on-manifold quality at later iters /
higher reveal fractions, not raw loss-space alignment. Otherwise
this regularizer is solving a non-bottleneck.

### 3.4 i_ref-biased sampler (T2; landed 2026-05-12)

Motivation: `design.md` §6 M1.5 T2 + §7 Q8. Training enumerates all
`(i_ref, i_target)` pairs with `i_ref < i_target < n_passes_actual`
(`dataset.py:188-203`). With `MAX_ITER = 6`, that's 15 pairs/block,
only 5/15 (33%) of which have `i_ref = 0`. Inference uses
`i_ref = 0` ~80–95% of the time (`hybrid_runner.py:178-258`, modulo
rollback rate). The model spends ⅔ of its gradient on pairs almost
never seen at inference.

Two new surfaces:

- `TataDeltaDataset.compute_index_weights(*, ref0_weight_multiplier)`
  → `torch.Tensor[len(index)]` — per-index weight for use with
  `WeightedRandomSampler`. Currently the only knob: pairs with
  `i_ref == 0` get `ref0_weight_multiplier` weight, all others get
  `1.0`.
- `train.py` builds a `torch.utils.data.WeightedRandomSampler` from
  the weights when `cfg.data.ref0_weight_multiplier != 1.0`, with
  `num_samples = len(train_ds)` and `replacement=True`. Falls back
  to the standard `shuffle=True` loader otherwise (pre-T2 behavior).

Sampling probability at multiplier `r`, given 5 `i_ref=0` pairs and
10 `i_ref>0` pairs per block (so weight mass `5r` vs `10`):

| `r` | P(`i_ref=0`) | comment |
|---:|---:|---|
| 1 (default-off) | 5/15 ≈ 33% | trial-2 baseline |
| 2 | 10/20 = 50% | mild bias |
| 3 (M1.5 default) | 15/25 = 60% | recommended start |
| 4 | 20/30 ≈ 67% | |
| 6 | 30/40 = 75% | |
| 8 | 40/50 = 80% | near inference rate |

Default `cfg.data.ref0_weight_multiplier = 3.0`. Tune upward if
`val/mse_by_gap_1` (i_ref=0 → i_target=1 pairs are most over-sampled
under T2 + biased toward i_ref=0) bottoms out while higher-gap bins
keep moving. Tune downward if generalization on i_ref≥1 pairs gets
visibly worse (rare-but-real inference regime when rollback fires).

Resume note: the sampler is rebuilt fresh from `cfg.data.*` at every
launch — no sampler state in the checkpoint. Reproducibility-wise,
the sampler uses `torch.manual_seed(cfg.seed)` set at startup.

---

## 4 · Training (`train.py`)

Single-file training loop, AdamW + cosine LR + warmup. Wandb logging.

Wallclock timers (`StepTimers`) wrap each section: `data / h2d / fwd /
loss / bwd / opt / val`. Prints `[time]` line every `log_every` steps.
CUDA-synced. After preload landed, `data` is typically <5%.

`run_val` mirrors the train step but inside `@torch.no_grad()`, also
bins `mse` by `(i_target − i_ref)` gap and by `reveal_frac`.

Mid-train GSM8K eval every `cfg.log.gsm8k_every` steps on a small
subset (`cfg.log.gsm8k_subset` problems), with `per_pos_threshold`
from `cfg.log.gsm8k_per_pos_threshold` (default 0.85).

### 4.1 Checkpoint format

`{"step": int, "model": state_dict, "opt": state_dict, "cfg": dict}`.
Saved every `cfg.checkpoint.every` to `cfg.checkpoint.out_dir`; only
`cfg.checkpoint.keep_last` most recent are kept.

State_dict shapes have changed across §1.5 (RoPE), §1.6 (SwiGLU), §3.2
(per-pos conf). Pre-§3.2 checkpoints **will not load**; mid-trial resumes
are fine.

### 4.2 Config — two M1 trials

Two config files live side-by-side; pick the one matching the trial:

| file | role |
|---|---|
| `configs/m1_llada_variant_c.yaml` | M1 trial-2 baseline of record. `mse_space=raw`, `λ_mse=0.1`, `dropout=0.0`, no `ref0_weight_multiplier` (uniform sampler). Lands in `ckpts/m1_llada_variant_c/`, wandb group `M1-llada`. |
| `configs/m1_5_llada_variant_c.yaml` | M1.5 Tier-1 trial. `mse_space=final_norm`, `λ_mse=1.0`, `dropout=0.1`, `ref0_weight_multiplier=3.0`. Lands in `ckpts/m1_5_tier1_llada_variant_c/`, wandb group `M1.5-tier1-llada`. |

Shared key knobs:

```
data.batch_size: 256          # drop to 128 on tight VRAM
data.preload: true            # ~80 GiB RAM
data.num_workers: 0           # forced by cluster shm cap
data.pin_memory: false        # same
model.d_model: 4096
model.n_layers: 2
model.d_ff_inner: 10944       # SwiGLU; ≈ 8/3 · d_model
optim.lr: 1.0e-4
optim.max_steps: 20000
log.gsm8k_per_pos_threshold: 0.85   # T4 sweep may surface a better default
```

To compare trial-2 baseline vs Tier-1, run each config end-to-end and
compare in wandb (separate groups) or via `eval_results/*.json`. The
caches are shared (`cache_v1/llada/`); only the model state diverges.

---

## 5 · Inference (`inference/hybrid_runner.py`)

`generate_with_delta(model, delta_model, final_norm, lm_head, token_embed,
prompt, ...)` → `(final_token_ids, stats)`.

Per block:

```
pass 0:    full backbone forward → h_ref, prefix_kv, past_key_values_to_s
           token transfer on full-seq logits, restricted to [s:e]
iter ≥ 1:
    if force_rollback_next:
        # partial-forward rollback (matches collect's iter ≥ 1)
        out = model(x[:, s:], past_key_values=past_key_values_to_s, use_cache=True)
        h_ref ← out.h[block region]
        past_key_values / prefix_kv unchanged (x[:, :s] hasn't changed within the block)
    delta_h, c_pos = delta_model(h_ref, prev_emb, prefix_kv, block_start_pos, prefix_kv_pad_mask)
    h_pred = h_ref + delta_h
    logits = lm_head(final_norm(h_pred))
    transfer_blk     ← Fast-dLLM's per-position transfer selection on `logits`
    per_pos_pass     ← c_pos ≥ per_pos_threshold
    agreement_blk    ← transfer_blk ∧ per_pos_pass
    commit only agreement_blk positions
    if |transfer_blk| > |agreement_blk|: force_rollback_next = True (§3.2)
```

Stats: `rollbacks`, `backbone_forwards`, `delta_forwards`, `disagreements`,
`per_block_passes`, `walltime`.

### 5.1 Agreement decoding rationale

Block-aggregate confidence is a poor fit for Fast-dLLM's per-position
commit semantics — a sharp-but-wrong delta can put high local confidence
on the wrong token at a single position, and Fast-dLLM commits it even
when the block-average looks fine. Agreement decoding gates per-position
on the delta model's own per-position confidence head, then forces a
rollback when even one position disagrees. Closes the
aggregate-vs-per-position signal mismatch identified in the post-mortem.

---

## 6 · The §3.1 alignment chain (decisions log)

This was load-bearing and went through several incorrect-then-corrected
framings. Recording the final state and how we got there.

**Final state:** collect and inference both use Fast-dLLM's
prefix-cache decoding at iter ≥ 1. Specifically:
`model(x[:, s:], past_key_values=past_to_s, use_cache=True)` with
auto-derive RoPE positions. The hidden states produced are *not*
equivalent to a hypothetical full forward at the same `x` state (T4
measures max abs diff ~6.0 / 1.8% relative) — but this is the
intentional Fast-dLLM behavior and is what the baseline
`generate_with_prefix_cache` produces. Training the delta model
against this distribution keeps it aligned with the eval baseline.

**Path we took to get here:**
1. Initial collect was correct (partial forward at iter ≥ 1, matching the baseline). First M1 training run produced GSM8K 0.72→0.10 with this collect.
2. T4 (`test_partial_full_forward_equivalence`) measured the 6.0 divergence between partial and full forward. **Claude misframed this as a position-id bug** and switched collect to full-forward-at-every-iter as a "fix."
3. The user pushed back: multi-token decoding has no full-forward baseline; Fast-dLLM is the canonical baseline and it's partial-forward by design. Reverted collect to partial-forward.
4. While investigating, surfaced a *real* bug: hybrid runner's rollback path was using full forward, producing h_ref from a different distribution than the delta model was trained on. Fixed: rollback now uses partial forward with the pass-0 cache.

**T4's purpose now:** documentation, not a gate. Prints "✓ EXPECTED — both partial paths diverge from full" → confirms the Fast-dLLM behavior. If a future LLaDA build makes the paths agree, T4 prints "⚠ NOTABLE" (informational; we could re-enable some optimizations).

**`replace_position` API:** Fast-dLLM v1 modeling exposes a separate
in-place-replace API meant for `generate_with_dual_cache`. We don't
use it. Our path matches `generate_with_prefix_cache`, which uses the
standard `past_key_values=past_to_s` + auto-derive RoPE.

---

## 7 · Eval (`eval/gsm8k_e2e.py`)

`run_gsm8k_eval(...)` runs hybrid + vanilla `generate_with_prefix_cache`
on the same GSM8K slice. Reports per-`per_pos_threshold`:

- `accuracy_hybrid`, `accuracy_vanilla`, `accuracy_delta`
- `speedup_ratio` (vanilla wall-time / hybrid wall-time)
- `mean_rollbacks`, `mean_backbone_calls` per problem
- `mean_disagreements` (§3.2) — proxy for how strict the per-pos gate is

Standalone CLI:
```
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/.../step_NNNN.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95
```

Decoding mode (`--factor` / `--threshold`) **must match what collect used** or
the delta is being tested in a regime it never trained on.

---

## 8 · Sanity tests (`sanity/`)

| # | test | what it checks | when to run |
|---|---|---|---|
| T1 | `python3 -m py_compile delta_model/...` | syntax / imports across the package | before any sit-and-wait command |
| T2 | `test_zero_init` | DeltaHead zero-init invariant; full forward signature + per-position loss path; new RoPE / RMSNorm / SwiGLU primitives don't crash | after model or loss changes |
| T3 | `test_collect_roundtrip` | cache schema matches `schema.py`; field shapes / dtypes correct; optional `prefix_kv_pad_mask` validated if present | after any collect run |
| T4 | `test_partial_full_forward_equivalence` | documents the Fast-dLLM partial-forward behavior (B1 & B3 both diverge from full forward → ✓ EXPECTED on this LLaDA build) | informational; run after any LLaDA upgrade or modeling-side change |
| diag | `inspect_llada_modeling` | dumps `LLaDAModelLM.forward` / `LLaDAModel.forward` signatures + RotaryEmbedding source + attention method source + targeted grep hits | when investigating an LLaDA modeling question |

---

## 9 · Active improvements / open items

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

### Performance / IO
- `[x]` Drop fp32 upcast in dataset (returns fp16; train casts to bf16 on GPU).
- `[x]` Move `token_embed` lookup to GPU post-batch (dataset returns `substituted_ids`).
- `[x]` Larger per-worker sample LRU (was 4, now 32). Mostly moot after preload.
- `[x]` Shard packing (`data/repack.py` + dataset auto-detect). Useful only if cache > RAM.
- `[x]` Preload-all-samples into RAM at `__init__` — the actual fix for the data-bottleneck.
- `[!]` Raise `/dev/shm` to re-enable workers — cluster-admin action; would give 3–5× on `data` on top of preload.

### Model / loss
- `[x]` RoPE on absolute positions (§1.5).
- `[x]` RMSNorm + SwiGLU primitives matched to backbone (§1.6).
- `[x]` Per-position confidence head + agreement decoding.
- `[x]` Fix BCE NaN: clamp `c_label_per_pos ∈ [0, 1]` (fp32-softmax roundoff).
- `[x]` **T1 (M1.5)** MSE-space switch (`raw` | `final_norm`) in
  `composite_loss`. M1.5 config uses `final_norm` with `λ_mse=1.0`.
  See §3.2. **Mechanically correct, did not move val KL — not the
  binding constraint at trial-2 data scale.**
- `[x]` **T2 (M1.5)** i_ref-biased `WeightedRandomSampler`
  (`data.ref0_weight_multiplier`, default 3.0). See §3.4. Tune
  upward to 4–8 in M2 if per-bin diagnostic supports it.
- `[x]` **T3 (M1.5)** `model.dropout = 0.1` (config-only). Reduced
  train-val KL gap 3.93× → 3.19× — directionally right but
  undersized. Try 0.2–0.3 in next round.
- `[~]` **T4** (eval-only) `per_pos_threshold` sweep on the M1.5
  checkpoint — `eval/gsm8k_e2e.py` already supports it; no code
  change. Free; run alongside M1.6 / T5. Command in `usage.md`.
- `[~]` **M1.6 = T6** `λ_mse = 0` ablation — own config
  (`m1_6_llada_variant_c.yaml`). Uses existing `cache_v1/llada/`.
  Parallel to M2 collect. Tests whether MSE is doing any load-bearing
  work after the M1.5 reallocation didn't move val KL.
- `[~]` **T5 (M2) — PRIMARY** scale data 5000 → 15000–20000 prompts
  into `cache_v1_20k/llada/`. Pure collect cost, no code change.
  Direct response to the block-as-sampling-unit overfitting
  signature (M1.5 status snapshot).
- `[ ]` **T7 (M2)** Deeper delta model (2 → 4 layers). Only if T5
  plateaus below 0.60 and per-bin diagnostic implicates capacity.
- `[ ]` **T8 = old C (M2)** Sliced 1-D distribution-matching
  regularizer on `h_pred` vs `h_target` — SIGReg mechanism. See §3.3.
- `[ ]` Variant A (cross-attn anchor) — M2 candidate.
- `[ ]` Variant B (AdaLN-Zero) — M2 candidate.
- `[ ]` Multi-layer hidden fusion (EAGLE-3 hint) — M2; requires recollect.

### Data
- `[x]` Schema v2: `prefix_kv` field, `prefix_window=64` stored, optional pad mask (§3.4, §3.5).
- `[ ]` Locality-aware sampler — only needed if cache > RAM.
- `[ ]` Per-iteration revealed-token snapshots in cache — flagged as "future collector additions" but not specified.

### Inference / correctness
- `[x]` §3.1 alignment chain audited; hybrid runner rollback uses partial forward.
- `[x]` §3.2 agreement decoding.
- `[ ]` Sample-level diagnostic: log per-position c_pos vs actual top-1 correctness at val time. Would tell us whether the conf head is well-calibrated independently of the per_pos_threshold setting.

### Accepted limitations (not targeted)
- **Trajectory drift / exposure bias.** Once delta commits a wrong token, `x[:, s:e]` diverges from any backbone-only trajectory. Subsequent `prev_emb` embeds the wrong token, compounding. Mitigation = train a good delta; no data-augmentation or scheduled-sampling planned for M1.
- **Greedy amplification.** `temperature=0` makes top-1 flips sensitive to small KL drift. Will resolve as quality improves.

---

## 10 · Cluster / infra notes

Recorded in `~/.claude/projects/.../memory/project_cluster_shm.md`:

- HPC node has `/dev/shm = 64 MiB` (Docker default). PyTorch DataLoader workers SIGBUS on shm pressure even with `set_sharing_strategy("file_system")` (the file-system strategy moves *tensor* data off shm, but torch.multiprocessing still uses shm for sync primitives). Workaround: `num_workers=0`. Real fix is to raise the container `--shm-size`.
- Fp16 hidden states from collect on this hardware peak at ~334 in absolute value. fp16 noise tolerance: ~5e-3 absolute.
- Trial-1 data point: 0.72 baseline GSM8K, 0.30 at 5k steps, 0.20 at 15k.

---

## 11 · Open research questions

(Restated from `design.md` §7 for context; engineering implications.)

1. **Is the per-position conf head well-calibrated?** Required for §3.2 agreement decoding to be a meaningful gate. M1 trial-2 measurement.
2. **What range of `per_pos_threshold` is useful?** Sweep at eval time. Too low → too many wrong commits, GSM8K drops. Too high → too many rollbacks, no speedup. Inflection point characterizes how confident the model deserves to be.
3. **Do later passes (i ≥ 3) need a different treatment?** Empirical shared-mass plateaus at ~58–60% for pass i ≥ 3. Possibly the delta model can't recover this gap and we should always rollback after pass 2.
4. **How do (i_ref, i_target) gap and reveal_fraction interact with accuracy?** Currently logged in val metrics binned by both — but not yet looked at as a diagnostic.
5. **Right loss space (T1 result: not the binding constraint).** M1.5 val KL identical to trial-2's despite swapping MSE-space, so loss-space reallocation alone does not close the train-val gap at this data scale. Open: M1.6 (T6 `λ_mse = 0`) will tell us whether MSE is doing *any* load-bearing work, or whether the 2-term loss (KL + BCE) matches M1.5 exactly.
6. **What is the actual inference `i_ref` distribution?** §3.4 T2 sampler is at multiplier 3 → 60% P(i_ref=0). Inference's actual distribution is unmeasured. Quick instrumentation: have `hybrid_runner.py` record `(i_ref_used, i_target)` tuples per delta forward and emit a histogram into `stats`. Land alongside the M1.6 eval. If inference is 90%+ i_ref=0, bump multiplier toward 8 in T5.
7. **Is trajectory drift the eval bottleneck (M2 gate for T8)?** §3.3's anti-drift regularizer is only worth implementing if mid-iter `h_pred` distribution diverging from `h_target` is what costs accuracy at later iters / higher reveal fractions. Diagnostic: log per-bin val-loss + shared-mass against `(i_ref, i_target)` gap after T5 lands.
8. **Block as effective sampling unit.** M1.5 overfitting onset at ~9.6 block-views says the 15 pairs/block share too much structure to count as independent examples. Open: should the dataset's sampling weight further down-weight intra-block redundancy, or is "one block ≈ one example" the right mental model and data-scaling (T5) the only fix? T5 is the experiment that answers this — if val KL drops materially with 3–4× more blocks, the model was data-starved at the block level.
