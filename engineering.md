# tata — engineering reference

For an agent or developer who is going to read the code and continue
the work. Organized by *surface area* (data / model / loss / training /
inference / eval / sanity tests), with each section laying out (a) the
current contract, (b) recent decisions, and (c) open items.

For *why* any of this exists, see `design.md`.
For *how* to run anything, see `usage.md`.

---

## Status snapshot (2026-05-14)

Current focus: **the shard-mode data loader.** The three-way M1.5
comparison (5k preload / ~11.8k BlockShardSampler / ~11.8k preload)
proved (a) data scaling *works* — preload+WRS at ~11.8k hits val/kl
**0.280** vs the 5k run's 0.293 — and (b) `BlockShardSampler` was the
cause of the m2_t5 / m1_5_data20k underperformance (val/kl 0.332),
because it draws a whole batch from one shard. `preload=true` is
RAM-capped at ~11.8k, so `InterleavedShardSampler` (§3.6) is the
unlock to scale the cache further. Next run:
`m1_5_data20k_interleaved_llada_variant_c.yaml` — should match the
preload run (~0.280) and confirm the sampler is the fix.

| area | state |
|---|---|
| Data schema | v2 (`prefix_kv` field, `prefix_window=64` stored, optional `prefix_kv_pad_mask`). Trial-2 / M1.5 cache reused for M1.6. T5 collect lands in a new dir. |
| Model | VariantC with RoPE on absolute positions (§1.5), RMSNorm + SwiGLU primitives (§1.6), per-position confidence head, dropout knob exposed (T3, §2.5). |
| Training | Full-train-split training + `test/`-split validation (`data.val_split`, §1.3). Single-process loader. Sampler: `InterleavedShardSampler` for shard caches (§3.6), i_ref-biased `WeightedRandomSampler` for preloaded (§3.4, T2). |
| Loss | Composite MSE + KL + BCE. MSE space configurable via `loss.mse_space` (`raw` legacy, `final_norm` recommended; §3.2). BCE target clamped to [0, 1]. |
| Inference | Hybrid runner: full forward @ pass 0, delta @ iter ≥ 1, partial-forward rollback (matches collect's iter ≥ 1). Agreement decoding via per-position threshold. |
| Eval | gsm8k_e2e harness sweeps `per_pos_thresholds`; reports hybrid/vanilla accuracy + speedup + disagreement count. |

Trial results history:
- **Trial 1** (pre-3.1 rollback fix, block-aggregate conf): 0.72 baseline → 0.20 at 15k.
- **Trial 2** (post-3.1 alignment audit, agreement decoding, BCE clamp): GSM8K **0.44** final. Train MSE/KL/BCE 12.6/0.054/0.29 vs val 16.1/0.38/0.59 — train-val KL gap ~4× hypothesized to be loss-space misallocation.
- **M1.5 Tier-1** (T1 final_norm MSE + T2 sampler + T3 dropout): final train 0.989/0.092/0.348, val 1.081/0.293/0.472. **GSM8K peak 0.40 @ step 5k, final 0.28 @ step 20k.** Val KL identical to trial-2 (0.293 vs 0.295); KL gap narrowed 3.93× → 3.19× (T3 working, undersized). Loss-space hypothesis **rejected as primary lever**: model peaks fast then memorizes. Overfitting onset at ~1.5k steps = ~9.6 block-views — at this data scale, the 15 pairs/block share enough structure that the *block* is the effective sampling unit, putting a 500M-param model at ~10 block-epochs of effective data by 1.5k.
- **M1.6 (T6, λ_mse = 0)**: final train 1.72/0.077/0.33, val 1.59/0.31/0.49. **GSM8K peak 0.36 @ step 10k, final 0.30 @ step 20k.** Train KL fits 0.012 better than M1.5, val KL is 0.02 worse — classic regularizer-removed signature, confirming **MSE was a soft regularizer, not noise**. Peak shifted later (10k vs 5k) and lower (0.36 vs 0.40). Decision: keep MSE but reduce weight to `λ_mse = 0.5` in T5 so MSE doesn't dominate (was ~60% of M1.5's weighted val loss).

Recent changes log (most recent first):

- **2026-05-15** Rollback semantics fixed in `inference/hybrid_runner.py`
  (§5.1). A rollback now commits tokens via vanilla Fast-dLLM on the
  backbone's own logits (ungated — the confidence gate only judges the
  *delta* model), guaranteeing ≥ 1 token of progress per rollback. Was a
  livelock at high `per_pos_threshold`: rollbacks committed nothing,
  blocks spun to `loop_cap` (0.95-sweep: `mean_rollbacks 467`,
  `speedup 0.10`, `accuracy 0.03`).
- **2026-05-15** `data/thin_cache.py` added — in-place, resumable cache
  thinner (drop block 7, cap iterations to 5, shrink prefix-KV window
  64→32; CLI-tunable). Thins `train/` only by default so `test/` stays
  full-resolution and validation/GSM8K compare every run on an identical
  held-out set. `data/repack.py` gained `--rm_source` (unlink per-sample
  files as each shard lands → peak disk ~1× not ~2×). `dataset.py` now
  reads the block count per-sample (not `S.NUM_BLOCKS`); `schema.py` /
  T3 validate against `meta`-recorded `num_blocks` / `max_iter`.
- **2026-05-14** `InterleavedShardSampler` landed in `data/dataset.py`
  (§3.6) — replaces `BlockShardSampler` as the default shard-mode sampler
  (`data.shard_sampler`, default `interleaved`). Keeps `active_shards`
  shards resident and mixes draws across all of them, so a batch spans
  many shards instead of one. Root-caused from the three-way M1.5
  comparison: same recipe + same ~11.8k cache, BlockShardSampler val/kl
  0.332 vs preload+WRS 0.280; the spsv 4096-vs-512 ablation was null
  because both confine a batch to one shard.
- **2026-05-14** `train.py` `data.val_split` knob (§1.3): default `test`
  validates on the hash-stable cache `test/` split and trains on the
  *full* train split; `holdout` is the legacy carve-`val_frac`-from-train
  behaviour. Makes val/kl comparable across cache sizes (the old
  `make_train_val_filter` carved a *different* 10% out of each
  differently-sized cache). All pre-2026-05-14 configs pinned to
  `val_split: holdout` (+ the four shard configs to `shard_sampler:
  block`) so their recorded results stay reproducible.
- **2026-05-12** `BlockShardSampler` landed in `data/dataset.py` (§3.5). Auto-selected when `preload=False` + shard mode (T5 default). Keeps memory bounded at ~4 GB for the 400-GB T5 cache by visiting one shard for `samples_per_shard_visit` consecutive indices.
- **2026-05-12** `train.py` gains best-checkpoint tracker (`_BestTracker`); saves `best_val_kl_step<N>.pt` + `best_gsm8k_step<N>.pt` independent of `keep_last` rotation. Sidecar `best_metrics.json` for resume.
- **2026-05-12** M1.6 result recorded; ranking finalized: T5 unambiguous primary, M1.6 keeps MSE but at half weight.
- **2026-05-12** `m2_t5_llada_variant_c.yaml` drafted: `λ_mse 0.5`, `dropout 0.2`, `weight_decay 0.05`, `gsm8k_every 2000`, reads `cache_v1_20k/llada/`.
- **2026-05-12** `eval/plot_metrics.py` added; `train.py` writes JSONL mirror of every wandb.log call to `<ckpt_dir>/metrics.jsonl`.
- **2026-05-12** `eval/gsm8k_e2e.py` `--out_json` auto-mkdirs.
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
      plot_metrics.py                       — matplotlib plotter for `<ckpt_dir>/metrics.jsonl` (multi-run compare)
    sanity/
      test_zero_init.py                              — T2: model+loss invariants
      test_collect_roundtrip.py                      — T3: cache schema check
      test_partial_full_forward_equivalence.py       — T4: documents Fast-dLLM partial-forward behavior
      inspect_llada_modeling.py                      — diagnostic for the loaded LLaDA modeling code
    configs/
      m1_llada_variant_c.yaml               — M1 trial-2 baseline (raw MSE, dropout 0, uniform sampler)
      m1_5_llada_variant_c.yaml             — M1.5 Tier-1 (T1+T2+T3) — final_norm MSE λ=1, dropout 0.1, ref0-biased sampler
      m1_6_llada_variant_c.yaml             — M1.6 = T6 (λ_mse = 0 ablation) — KL+BCE only loss; rest matches M1.5
      m2_t5_llada_variant_c.yaml            — M2 T5 (data 20k) — λ_mse 0.5, dropout 0.2, wd 0.05, gsm8k_every 2k
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
`shard_lru_max` overrides the shard-mode LRU cap — `train.py` sets it to
`active_shards + 1` so `InterleavedShardSampler`'s window stays resident
(§3.6).

**Validation source (`data.val_split`, 2026-05-14).** Default `test`:
`train.py` builds the train dataset from the *full* `train/` split (no
`index_filter`) and the val dataset from the cache's `test/` split — the
800 hash-stable prompts, identical across cache sizes, so val/kl is
comparable run-to-run. `holdout` restores the legacy
`make_train_val_filter` carve (a `val_frac` slice of `train/`, keyed on
positional `s_idx`, which makes the held-out set differ between
differently-sized caches). `make_train_val_filter` is retained only for
`holdout`.

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
- **Locality-aware sampler** — landed 2026-05-12 as `BlockShardSampler`
  (§3.5). Auto-selected when `preload=False` + shard mode. Default for
  T5+ at 20k samples where the cache no longer fits in RAM.

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

### 3.5 BlockShardSampler — locality-aware sampling for large caches (T5+)

Motivation: at 5000 prompts the cache fits in RAM (`preload: true` ≈
80–100 GB resident, well under the 300 GB bound). At T5's 20000
prompts the cache is ~400 GB on disk — does NOT fit. We need
`preload: false` + shards, but a *random* shuffle over a shard-mode
dataset thrashes the LRU because each random batch hits ~B distinct
shards.

Solution: a sampler that groups consecutive accesses into the same
shard for `samples_per_shard_visit` indices before moving on. The
shard LRU (default 4 shards ≈ 4 GB) stays hot through the entire
visit; disk I/O drops from "1 shard load per batch" to "1 shard load
per (samples_per_shard_visit ÷ batch_size) batches".

Implementation: `BlockShardSampler` in `data/dataset.py`. Each round:

  1. Permute the shard order (seeded for reproducibility).
  2. For each shard, draw `samples_per_shard_visit` indices with
     `torch.multinomial(replacement=True)` using that shard's slice
     of the per-index weight tensor (so T2 weighting is preserved
     within each shard).
  3. Yield each index. Continue until `num_samples` total emitted.

Sampler selection in `train.py:main`:

  - `preload=False` + shard mode    → `BlockShardSampler` (auto)
  - `preload=True`  OR per-sample   → `WeightedRandomSampler` (if T2 active)
  - neither                          → standard `shuffle=True`

Config knob: `data.samples_per_shard_visit` (default 4096 = 16
batches at `batch_size=256`). Trade-off:

| spsv | batches/visit | I/O overhead (est) | gradient correlation |
|---:|---:|---:|---|
| 256 (=B) | 1 | ~50% | low (one batch per shard) |
| 1024 | 4 | ~15% | mild |
| **4096** | **16** | **~6%** | moderate, recommended default |
| 8192 | 32 | ~3% | high (long same-shard streaks) |

Note: shard locality and T2 weighting combine cleanly because each
shard contains roughly the same proportion of `i_ref=0` pairs (5/15
by structure), so per-shard weighted sampling preserves the global
target distribution.

End-to-end workflow for T5+:

  1. `collect_llada.py` writes per-sample (existing flow).
  2. `data/repack.py --shard_size 50` packs into shards + creates
     `shards_manifest.json`. Dataset auto-detects shard mode.
  3. Train config sets `data.preload: false`. `train.py` constructs
     a shard sampler from `data.shard_sampler` (§3.6).

### 3.6 InterleavedShardSampler — the default shard-mode sampler (2026-05-14)

**Why BlockShardSampler was retired as the default.** §3.5's design —
"keep consecutive accesses inside one shard" — means an entire batch is
drawn from a single shard (~`shard_size` ≈ 50 prompts). The three-way
M1.5 comparison made the cost unambiguous: same recipe, same ~11.8k
cache, only the loader differing —

| loader | train/kl | val/kl |
|---|---:|---:|
| `preload` + `WeightedRandomSampler` | ~0.15 | **0.280** |
| `BlockShardSampler` (spsv 512) | ~0.21 | 0.332 |

Single-shard batches give correlated, high-variance gradients → the
optimizer plateaus ~2× worse on *both* train and val. The spsv
4096-vs-512 ablation was null because every spsv confines a batch to one
shard — there is no spsv that fixes it.

**`InterleavedShardSampler`** (`data/dataset.py`) keeps a *window* of
`active_shards` shards resident at once and interleaves draws across all
of them, so each batch the DataLoader cuts spans ~`active_shards ·
shard_size` prompts. Per round it permutes the shard order, slides a
non-overlapping window of `active_shards` shards, pools each window's
indices + T2 weights, draws `chunk_size` indices
(`multinomial(replacement=True)`), shuffles them, and yields. The shuffle
is what mixes shards within every batch. I/O is identical to
`BlockShardSampler` (each shard loaded once per round); only the in-RAM
working set grows from 1 to `active_shards` shards — so the dataset's
shard LRU must be ≥ `active_shards` (`train.py` passes
`shard_lru_max = active_shards + 1`).

Config knobs (`data.*`, used when `preload: false` + shard mode):

| knob | default | meaning |
|---|---|---|
| `shard_sampler` | `interleaved` | `interleaved` \| `block` (legacy §3.5) |
| `active_shards` | `8` | shards resident & interleaved at once; RAM ≈ `active_shards · shard_bytes` |
| `shard_chunk_size` | auto | indices per window; auto = `active_shards · mean_pairs_per_shard` (≈ one round = one epoch) |

Sampler selection in `train.py:main`:

  - `preload=False` + shard + `shard_sampler=interleaved` → `InterleavedShardSampler` (default)
  - `preload=False` + shard + `shard_sampler=block`       → `BlockShardSampler` (legacy)
  - `preload=True`  OR per-sample, `ref0_mult ≠ 1.0`      → `WeightedRandomSampler`
  - otherwise                                            → standard `shuffle=True`

Sanity: `python -m delta_model.sanity.test_interleaved_sampler` (T5; no
cache / GPU needed) — checks exact count, per-batch shard spread,
coverage, and that T2 weights bias the draw.

---

## 4 · Training (`train.py`)

Single-file training loop, AdamW + cosine LR + warmup. Wandb logging
**plus** a local JSONL mirror at `<ckpt_dir>/metrics.jsonl` — every
`wandb.log` payload is appended as one JSON line (`{step, ...}`). This
gives full on-machine history independent of wandb's cloud, parseable
by `eval/plot_metrics.py` (matplotlib). Append-mode survives resumes.

Wallclock timers (`StepTimers`) wrap each section: `data / h2d / fwd /
loss / bwd / opt / val`. Prints `[time]` line every `log_every` steps.
CUDA-synced. After preload landed, `data` is typically <5%.

`run_val` mirrors the train step but inside `@torch.no_grad()`, also
bins `mse` by `(i_target − i_ref)` gap and by `reveal_frac`.

Mid-train GSM8K eval every `cfg.log.gsm8k_every` steps on a small
subset (`cfg.log.gsm8k_subset` problems), with `per_pos_threshold`
from `cfg.log.gsm8k_per_pos_threshold` (default 0.85).

### 4.1 Checkpoint format and best-tracking

Two parallel checkpoint streams live in `cfg.checkpoint.out_dir`:

**Rolling** — `step_<NNNNNNN>.pt` files (zero-padded to 7 digits).
Saved every `cfg.checkpoint.every` steps. Only `cfg.checkpoint.keep_last`
most recent are kept; older ones get unlinked on the next save.
Used for resume.

**Best-by-metric** — `best_val_kl_step<N>.pt` and
`best_gsm8k_step<N>.pt` (step not zero-padded). Maintained by
`_BestTracker` in `train.py`. At each val pass:
`val_kl` is compared against the running best (mode `"min"`); if it
improves, the prior `best_val_kl_step*.pt` is unlinked and a new one
written at the current step. Same for GSM8K mid-training eval with
`accuracy_hybrid` (mode `"max"`). NaN / None values are ignored.

The two streams are independent: `keep_last` rotation only touches
`step_*.pt`, so peaks captured by the best-tracker survive even when
the run continues into overfitting territory (the M1.5 lesson: peak
at step 5k got rotated away under `keep_last=3` by step 20k).

Resume semantics: `<ckpt_dir>/best_metrics.json` is a sidecar that
records the running best per metric (`{label: {value, step, mode}}`).
Loaded at `_BestTracker.__init__`, so resumes from `--resume_from`
don't lose history. If a resumed run produces a worse-than-prior
best, no file is written; if it produces a new best, the prior best
file is replaced.

Checkpoint dict contents:
```
step:        int
model:       state_dict
opt:         state_dict
cfg:         dict
# best_*.pt files only:
best_label:  str           # e.g. "val_kl" or "gsm8k"
best_value:  float
best_mode:   "min" | "max"
```

State_dict shapes have changed across §1.5 (RoPE), §1.6 (SwiGLU), §3.2
(per-pos conf). Pre-§3.2 checkpoints **will not load**; mid-trial resumes
are fine.

### 4.2 Config — four trials side-by-side

Four config files live in `configs/`; pick the one matching the trial.

| file | trial | loss | dropout | wd | ref0 mult | cache |
|---|---|---|---:|---:|---:|---|
| `m1_llada_variant_c.yaml` | M1 trial-2 baseline (GSM8K 0.44 final) | `raw`, λ=0.1 | 0.0 | 0.01 | — | `cache_v1/llada` |
| `m1_5_llada_variant_c.yaml` | M1.5 Tier-1 (peak 0.40 @ 5k, final 0.28) | `final_norm`, λ=1.0 | 0.1 | 0.01 | 3.0 | `cache_v1/llada` |
| `m1_6_llada_variant_c.yaml` | M1.6 (λ_mse = 0 ablation; peak 0.36 @ 10k, final 0.30) | `final_norm`, λ=0.0 | 0.1 | 0.01 | 3.0 | `cache_v1/llada` |
| `m2_t5_llada_variant_c.yaml` | M2 T5 (data 20k; in progress) | `final_norm`, λ=0.5 | 0.2 | 0.05 | 3.0 | `cache_v1_20k/llada` |

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
        # ROLLBACK — a pure backbone Fast-dLLM step (§5.1).
        out = model(x[:, s:], past_key_values=past_key_values_to_s, use_cache=True)
        h_ref ← out.h[block region]                  # anchor for the NEXT delta pass
        # vanilla Fast-dLLM transfer on out.logits — NO confidence gate;
        # commits ≥ 1 token, mirrors collect's iter ≥ 1 path exactly.
        commit Fast-dLLM's transfer set on out.logits
        force_rollback_next = False; continue         # next iter is a delta pass
    # ---- delta pass ----
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
`per_block_passes`, `per_block_revealed_at_finish`, `walltime`.

### 5.1 Agreement decoding + rollback semantics

Block-aggregate confidence is a poor fit for Fast-dLLM's per-position
commit semantics — a sharp-but-wrong delta can put high local confidence
on the wrong token at a single position, and Fast-dLLM commits it even
when the block-average looks fine. Agreement decoding gates per-position
on the delta model's own per-position confidence head, then forces a
rollback when even one position disagrees. Closes the
aggregate-vs-per-position signal mismatch identified in the post-mortem.

**A rollback is a vanilla Fast-dLLM step, not a gated one (2026-05-15).**
The per-position confidence gate is the *delta model's* self-assessment —
"can the small model match what LLaDA would commit here." It has no
meaning for a real backbone forward. So a rollback runs the partial
backbone forward and commits its tokens via plain Fast-dLLM
(`factor` / `threshold`), ungated — which commits ≥ 1 token. This
guarantees forward progress: a rollback always pays one backbone forward
*and* advances the block.

The previous code discarded `out.logits` on a rollback and re-gated the
refreshed `h_ref` through the delta model. At high `per_pos_threshold`
that livelocked — every rollback committed nothing, so a block spun to
`loop_cap` (~62 backbone forwards/block, 10× slower than vanilla, blocks
left unfinished → garbage output). Observed in a 0.95-threshold sweep:
`mean_rollbacks 467`, `speedup 0.10`, `accuracy 0.03`,
`mean_revealed_per_block 4.87/32`. With the fix, max rollbacks is bounded
at ~`block_length` per block (each commits ≥ 1 of the 32 positions), the
block always finishes, and high `per_pos_threshold` degrades gracefully
to ≈ the backbone-only baseline instead of livelocking.

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
- `[x]` **T4** (eval-only) `per_pos_threshold` sweep on the M1.5
  checkpoint — done, result pending discussion.
- `[x]` **M1.6 = T6** (`λ_mse = 0` ablation): MSE is a soft
  regularizer (train fits 0.012 better, val 0.02 worse). Don't drop
  entirely; T5 uses `λ_mse = 0.5` to keep MSE without dominating.
- `[x]` Best-checkpoint tracker (`_BestTracker` in `train.py`) —
  saves `best_val_kl_step<N>.pt` and `best_gsm8k_step<N>.pt` next
  to rolling `step_*.pt` files. Sidecar `best_metrics.json` for
  resume. See §4.1.
- `[x]` Local metrics JSONL mirror — `<ckpt_dir>/metrics.jsonl`,
  plotted by `eval/plot_metrics.py`. See §4 intro.
- `[~]` **T5 (M2) — PRIMARY** scale data 5000 → 20000 prompts
  into `cache_v1_20k/llada/`. Collect in progress. Training config
  `m2_t5_llada_variant_c.yaml` drafted: `λ_mse=0.5`, `dropout=0.2`,
  `weight_decay=0.05`, `gsm8k_every=2000`.
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
