# tata delta-model — running / smoke-testing

Operator's manual. Companion to `scoping.md` (design) and
`implementation_plan.md` (code structure). Use this doc when you're
about to run something on a GPU box.

`tata/` is **a standalone repo** — it does not import from any
sibling `peft_project/` tree. The only external code dependency is
**Fast-dLLM v1**, located at runtime via path / env var (see below).

---

## §3.1 fix recovery — RUN THIS FIRST

The §3.1 partial-forward `position_ids` bug was confirmed on 2026-05-11
(T4 reported `max=6.0`, `mean=0.089`, `rel=1.8%` — far above the 5e-3
fp16-noise tolerance). The fix landed in `collect_llada.py:332`, but the
existing cache was built before the fix and is poisoned. **All training
results to date are uninterpretable.** Sequential recovery plan, ~5–8 h
total wall time. Run on the cluster, in this order:

### 1. Verify the fix end-to-end (~30 s, GPU + Fast-dLLM)

```bash
python -m delta_model.sanity.test_partial_full_forward_equivalence \
    --fast_dllm_path external/Fast-dLLM/v1
```

T4 now runs both paths and reports each. Expected output:

```
[sanity] Path B1 — partial forward, NO position_ids (collect-pre-fix path):
[sanity] ✗ B1 vs A (full): max=6.0e+00 ...        ← still broken (expected)

[sanity] Path B2 — partial forward, WITH position_ids (collect-post-fix path):
[sanity] ✓ B2 vs A (full): max=~1e-3 ...           ← fix recovered equivalence

[sanity] ✓ FIX VERIFIED — without position_ids the partial forward is broken
(B1 diverges), but passing position_ids=arange(s, len) recovers full-forward
equivalence (B2 within tolerance).
```

If **B2 also fails** (max > 5e-3): STOP and report. Means LLaDA's modeling
needs more than `position_ids` (possibly `cache_position` or attention-mask
tweaks); we'd investigate before sinking 7 hours into a recollect.

### 2. Move the broken artifacts aside (5 s)

Don't delete — keep as a "broken-cache" reference for diff / postmortem.

```bash
mv cache_v1/llada                      cache_v1/llada_broken_3_1
mv ckpts/m1_llada_variant_c            ckpts/m1_llada_variant_c_pre_3_1   2>/dev/null || true
```

(The `2>/dev/null || true` on the ckpt dir tolerates "doesn't exist".)

### 3. (Optional but recommended) Smoke recollect to sanity-check the fix in collect (~3 min)

Uses 5 built-in prompts; no HuggingFace auth needed. Cheap insurance that
the fixed `collect_llada.py` doesn't crash and produces well-formed cache
files.

```bash
python -m delta_model.data.collect_llada \
    --prompts_file delta_model/data/sample_prompts.txt \
    --output_root cache_v1/llada_smoke_post_fix \
    --fast_dllm_path external/Fast-dLLM/v1

python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1/llada_smoke_post_fix/test/sample_*.pt"
```

Pass: roundtrip prints `5 OK, 0 bad.`

### 4. Real recollect (~4–7 h on a single A100/H100)

```bash
python -m delta_model.data.collect_llada \
    --n_train 5000 --n_test 800 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Resumable — if interrupted, just re-run; existing files are skipped.

### 5. Sanity-check the new cache (~10 s)

```bash
python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1/llada/test/sample_*.pt" 2>&1 | tail -5
```

Expected: `800 OK, 0 bad.`

### 6. Train fresh (~3–4 h for 20k steps)

No `--resume_from` — pre-fix checkpoints were trained on poisoned data
AND under a different architecture (pre §1.5). Doubly invalid.

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

What to watch in wandb:
- `train/loss` curve. Initial value should be lower than the pre-§3.1 run
  (h_target is now sane).
- `val/loss` should track `train/loss` for longer (no divergence at 8k).
- `gsm8k/accuracy_hybrid` per checkpoint: trend should be **rising** over
  training, not falling. Absolute number doesn't have to reach baseline
  0.72 yet — the *direction* tells us whether §3.1 was the dominant issue.

### 7. Mid-training decision point (around step 5k)

Check `gsm8k/accuracy_hybrid` in wandb at step 5000:
- **Higher than the pre-fix run's 0.30 at step 5000** → §3.1 was the
  dominant issue. Let training run to 20k.
- **Same or worse** → §3.1 wasn't the only issue. Stop the run and we
  move to §3.2 (per-position confidence + agreement decode) before any
  more training.

---

## Picking up after the §1 model+IO refactor

Pre-§3.1-recovery context. After step 6 above, the §3.1 row in this
table is resolved; the others stay relevant for any subsequent fresh run.

| | required? | command |
|---|---|---|
| **Re-collect data** | **YES** — see "§3.1 fix recovery" above. | step 4 in §3.1 recovery |
| **Re-run zero-init sanity** | **Yes** — exercises the new RoPE / RMSNorm / SwiGLU / `block_start_pos` plumbing. | `python -m delta_model.sanity.test_zero_init` |
| **Discard old checkpoints** | **Yes** — VariantC's state_dict shape changed (RMSNorm keys, SwiGLU keys, dropped `pos_emb` / `prefix_proj`). Don't `--resume_from` an M1-pre-§1.5 ckpt. | step 2 in §3.1 recovery |
| **Repack into shards** | Optional — pure shard repack alone does NOT fix random-shuffle data thrash; `data.preload: true` (default, see below) is the right call on big-RAM nodes. | `python -m delta_model.data.repack --cache_root cache_v1/llada` |
| **Use `data.preload: true`** | **Yes** — already the default in the config. Loads all samples into RAM at startup; eliminates per-step disk I/O. Needs ~80 GiB RAM at default cache size. Set `data.preload: false` only if RAM-constrained. | (no command — just leave the config alone) |
| **Update config** | Already done in this repo: `d_ff: 16384` → `d_ff_inner: 10944`. | (none) |
| **Audit collect for §3.1** | **DONE 2026-05-11** — bug confirmed, fix applied, recollect required (see "§3.1 fix recovery" above). | (resolved) |

---

## The cwd convention (read this first)

All commands below assume your working directory is **the root of
the tata repo itself**. After cloning, you should see:

```
$ cd <wherever>/tata
$ ls
delta_model/  scoping.md  implementation_plan.md  usage.md  ...
```

Internal imports are relative (`from .data import ...`,
`from ..llada_runtime import ...`), so the package is always called
`delta_model` regardless of where you cloned tata.

`python -m delta_model.X` resolves because `delta_model/` lives in
the cwd. If you want to run from a different cwd, do
`PYTHONPATH=/path/to/tata python -m delta_model.X` instead.

---

## Prerequisites — read before running anything

| Step | GPU? | HF login? | Fast-dLLM v1? | Time |
|---|---|---|---|---|
| 1. Compile-check | no | no | no | <5 s |
| 2. Zero-init sanity | no | no | no | ~5 s |
| 3. Smoke test (built-in prompts) | **yes** | no | **yes** | ~2-5 min |
| 4. Real collect (Nemotron) | **yes** | **yes** | **yes** | 4-7 h |
| 4b. Shard repack (optional) | no | no | no | ~5 min |
| 5. Train | **yes** | no | **yes**¹ | 3-4 h |
| 6. Eval | **yes** | no | **yes** | depends on N |

¹ training only needs Fast-dLLM v1 because we lift `final_norm` and
`lm_head` from the loaded LLaDA backbone.

### Where to put Fast-dLLM v1 on disk

The recommended layout (cwd = the tata repo root):

```
tata/
├── delta_model/                  ← all code here
│   ├── data/sample_prompts.txt
│   └── …
└── external/
    └── Fast-dLLM/
        └── v1/                   ← clone of github.com/NVlabs/Fast-dLLM v1
            ├── llada/
            │   └── model/
            │       └── modeling_llada.py
            └── dream/
```

Fast-dLLM v1 is NOT bundled. Get it once with:

```bash
mkdir -p external && cd external
git clone https://github.com/NVlabs/Fast-dLLM
cd ..   # back to the tata repo root
```

You don't have to put it under `external/` — it just has to be
findable. Three ways the runtime discovers it (in priority order):

1. CLI flag: `--fast_dllm_path external/Fast-dLLM/v1`
2. Env var: `export FAST_DLLM_V1_PATH=/abs/path/to/Fast-dLLM/v1`
3. Default: `external/Fast-dLLM/v1` (relative to cwd)

If Fast-dLLM is missing the runtime raises a clear `FileNotFoundError`
listing those three options.

### Python deps

```bash
pip install "torch>=2.4" transformers>=4.44 datasets h5py wandb pyyaml \
            numpy huggingface_hub
```

(LLaDA itself is downloaded from `GSAI-ML/LLaDA-8B-Instruct` on first
load — public, no auth needed for the model. HF login is only
required for the gated Nemotron Post-Training v2 dataset, used only
in the real collect step.)

### One-time auth

```bash
wandb login                 # before training (step 5)
huggingface-cli login       # before real Nemotron collect (step 4 only)
```

---

## Diagnostic tests reference

Quick-reference table of every check command, what it tests, and the
pass criterion. Run T1–T2 freely (no GPU needed). Run T4 before
trusting any training run that came out of the existing cache.

| # | Test | Purpose | Cost | Command |
|---|---|---|---|---|
| T1 | Compile-check | Catch syntax errors / broken imports across the package. | <5 s, no torch needed | see [Step 1](#1-compile-check-no-torch-needed) below |
| T2 | Zero-init sanity | Δh head is zero-init at step 0 (so MSE term equals the h_ref-reuse baseline) AND the new RoPE / RMSNorm / SwiGLU plumbing from §1.5+§1.6 actually runs end-to-end. | ~5 s, CPU-only | `python -m delta_model.sanity.test_zero_init` |
| T3 | Cache-format roundtrip | On-disk cache schema matches `data/schema.py`; per-block tensor shapes/dtypes correct; reveal pattern is monotone (positions never un-reveal). | ~10 s per sample | `python -m delta_model.sanity.test_collect_roundtrip "cache_v1/llada_smoke/test/sample_*.pt"` |
| T4 | **§3.1 partial-vs-full forward equivalence** | Verifies that `model(x[:, s:], past_key_values=cache_to_s)` (used by collect at iter ≥ 1 and by Fast-dLLM vanilla) produces the same block-region hidden states as `model(x)` at the same `x` state. If they diverge, RoPE position IDs aren't being auto-derived in partial-forward mode, which would mean every `h_per_pass[i ≥ 1]` in the cache is computed at wrong absolute positions — silent corruption of all training data. | ~30 s, **GPU + Fast-dLLM required** | `python -m delta_model.sanity.test_partial_full_forward_equivalence --fast_dllm_path external/Fast-dLLM/v1` |

### Pass / fail criteria

- **T1** prints `OK`. Anything else → fix the listed file before continuing.
- **T2** prints `delta_h.abs().max() = 0.000000e+00` AND `✓ zero-init sanity passed`. Failure → model rewrite (§1.5 / §1.6) broke the zero-init invariant; check `DeltaHead`.
- **T3** prints `[OK ]` for every sample, ending with `N OK, 0 bad.`. Bad samples have specific error lines describing the schema violation; fix the cache file or re-collect.
- **T4 PASS:**
  ```
  [sanity] |full - partial|: max = X.XXe-04  mean = Y.YYe-05  relative-to-max = Z.ZZe-04
  [sanity] ✓ PASS — partial and full forward agree to within 5e-03. §3.1 position-id alignment is fine.
  ```
  Means §3.1 is **not** the bug; the GSM8K e2e issue is something else (§3.2 calibration or trajectory drift). Existing cache is fine.
- **T4 FAIL:**
  ```
  [sanity] |full - partial|: max = X.XXe-01  …
  [sanity] ✗ FAIL — divergence X.XX e-01 > 5e-03.
  [sanity]   Consistent with the §3.1 position-id bug: the partial forward at iter ≥ 1 is rotating Q/K from absolute position 0 instead of S, so all `h_per_pass[i ≥ 1]` in the cache are computed at wrong positions.
  [sanity]   Likely fix: in `collect_llada.py:332`, pass
  [sanity]     position_ids=torch.arange(s, x.shape[1], device=x.device).unsqueeze(0)
  [sanity]   to the partial forward, then re-collect the cache from scratch.
  ```
  Means **the bug is real**. The current cache is poisoned and must be re-collected after the fix lands. All training results so far are uninterpretable.

### Recommended run order

1. **First time / after major code changes:** T1 → T2.
2. **After a `collect_llada` run:** T3 on the new cache.
3. **Before trusting any training run on the existing cache:** T4. This is the §3.1 verification; it's currently the most important diagnostic because it disambiguates "the model doesn't generalize" from "the cache is wrong."
4. **After T4 passes (or after re-collect if T4 failed):** proceed with training (Step 5 below).

---

## Build order — what to run, in order

All commands below assume cwd = **tata repo root**.

### 1) Compile-check (no torch needed)

```bash
python3 -m py_compile \
    delta_model/data/schema.py \
    delta_model/data/collect_llada.py \
    delta_model/data/dataset.py \
    delta_model/data/repack.py \
    delta_model/llada_runtime.py \
    delta_model/models/heads.py \
    delta_model/models/variant_c.py \
    delta_model/losses.py \
    delta_model/inference/hybrid_runner.py \
    delta_model/eval/shared_mass.py \
    delta_model/eval/gsm8k_e2e.py \
    delta_model/sanity/test_collect_roundtrip.py \
    delta_model/sanity/test_zero_init.py \
    delta_model/sanity/test_partial_full_forward_equivalence.py \
    delta_model/train.py \
  && echo OK
```

Pass criterion: prints `OK` (no syntax errors).

### 2) Zero-init sanity (CPU-only, no Fast-dLLM, no LLaDA)

Confirms the model + loss invariants before sinking any GPU time. Also
exercises the §1.5/§1.6 plumbing: RoPE on absolute positions, RMSNorm,
SwiGLU FFN, and the new `block_start_pos` argument.

```bash
python -m delta_model.sanity.test_zero_init
```

Pass criteria:
- prints `delta_h.abs().max() = 0.000000e+00` (Δh head is still zero-init).
- prints `✓ zero-init sanity passed`.
- The MSE term must equal `((h_target - h_ref) ** 2).mean()` exactly
  (the "h_ref reuse" baseline) and KL must be ≥ 0.

If this fails after the §1.5/§1.6 refactor, the most likely culprits
are: RoPE buffer dtype getting downcast (we recompute fp32 on the fly to
avoid this), `block_start_pos` shape mismatch, or `_split_prefix_kv`
expecting `n_kv_heads * d_head == d_model`.

### 3) Smoke test (GPU + Fast-dLLM, no HF login)

Runs the full collect pipeline on 5 built-in prompts. **No HuggingFace
auth required** because `--prompts_file` bypasses Nemotron. Verifies
the LLaDA + Fast-dLLM + cache-write path end to end.

```bash
python -m delta_model.data.collect_llada \
    --prompts_file delta_model/data/sample_prompts.txt \
    --output_root cache_v1/llada_smoke \
    --fast_dllm_path external/Fast-dLLM/v1

python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1/llada_smoke/test/sample_*.pt"
```

Pass criteria:
- collect prints `[collect] decoding mode: factor=1.0` once at startup
  (this is the default; pass `--threshold 0.9` instead if ablating).
- collect prints `[collect] N test file id=… done in YY.Y s` for each
  of the 5 prompts; no `FAILED` lines.
- roundtrip prints `[OK ]` for every sample, ending with `5 OK, 0 bad.`
- inspect a shard to confirm the decoding metadata is recorded:
  ```bash
  python -c "import torch; s = torch.load(sorted(__import__('glob').glob('cache_v1/llada_smoke/test/sample_*.pt'))[0]); \
  print('meta.decoding =', s['meta']['decoding']); \
  print('n_passes_actual per block =', [b['n_passes_actual'] for b in s['blocks']])"
  ```
  Expect `meta.decoding == {'mode': 'factor', 'factor': 1.0}` and most
  blocks at `n_passes_actual ≤ MAX_ITER=6`. If many blocks hit 6
  exactly, the trajectory is being truncated — investigate.
- inspect the manifest to confirm the same mode is pinned cache-wide:
  ```bash
  python -c "import json; m = json.load(open('cache_v1/llada_smoke/manifest.json')); print(m['decoding'])"
  ```

### 4) Real data collection (GPU + Fast-dLLM + HF login)

Produces `cache_v1/llada/{train,test}/sample_XXXXXXXX.pt` plus
`cache_v1/llada/manifest.json`. The test split is **hash-stable** —
re-running with a larger `--n_train` will not touch the same 800 test
samples, so you can grow the train set across runs without
contaminating eval.

```bash
huggingface-cli login    # one-time, accept Nemotron license

python -m delta_model.data.collect_llada \
    --n_train 5000 --n_test 800 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Optional knobs:
- `--subset_ratios '{"chat":0.4,"math":0.4,"code":0.2,"stem":0.0}'`
  — math-heavier mix.
- `--factor 1.0` (**default**) — Fast-dLLM dynamic per-rank threshold,
  the paper-recommended parallel-decoding mode. Per-rank bar is
  `1 - factor/(rank+1)` so rank-1 is always committed and the bar
  tightens for lower-confidence ranks.
- `--threshold 0.9` — switches to the fixed-threshold variant
  (`get_transfer_index`) instead. Mutually exclusive with `--factor`.
  Use only if eval will also be run in fixed-threshold mode — the
  training and inference decoding modes **must match** or the
  `(i_ref, i_target, reveal_frac)` distribution will diverge.
- `--start_index N` to resume a partially-done run.

Storage: ~80 GB at default 5800 samples (5000 train + 800 test).

### 4b) Shard repack — usually NOT what you want

Important context first: with `batch_size=256` and shuffled access across
thousands of samples / hundreds of shards, **a batch touches ~250 unique
files no matter the layout**. An LRU of any practical size hits almost
nothing, so shard packing on its own doesn't fix data-load thrash. The
fix that actually works on big-RAM nodes is `data.preload: true` (the
default in this repo) — see Step 5's note about the `[dataset] preloaded
N/M samples` line at startup.

Shard repack is still recorded here for two cases where it does help:
1. cache larger than RAM, so preload isn't feasible — shards reduce
   per-file syscall / inode overhead and let `torch.load` do bigger
   sequential reads;
2. paired with a locality-aware sampler that keeps consecutive batches
   reading from the same shard.

If you've decided one of those applies, run:

```bash
python -m delta_model.data.repack \
    --cache_root cache_v1/llada \
    --shard_size 50 \
    --output_subdir shards
```

Outputs:
- `cache_v1/llada/shards/{train,test}/shard_NNNNN.pt` — packed shards.
- `cache_v1/llada/shards_manifest.json` — index the dataset uses to
  discover shards. The dataset auto-detects this file at startup and
  switches to shard mode (no code change needed). Delete the manifest
  to revert to per-sample mode.

Non-destructive: the original per-sample files are kept. Shard build is
single-pass, ~5 min on a fast disk.

Verify after repack (no GPU needed):

```bash
python -c "
import json, torch
m = json.load(open('cache_v1/llada/shards_manifest.json'))
print(f'{len(m[\"shards\"])} shards, sizes:',
      sorted({r[\"n_samples\"] for r in m[\"shards\"]}))
s = torch.load(sorted(__import__('glob').glob('cache_v1/llada/shards/train/shard_*.pt'))[0],
               weights_only=False)
print('shard_meta:', s['shard_meta'])
print('samples in first shard:', len(s['samples']))
print('first sample keys:', list(s['samples'][0].keys()))
"
```

Expect: shard count ≈ ceil(N_samples / shard_size), `n_samples` mostly
== shard_size with one tail shard ≤ shard_size, and per-sample keys
matching the un-sharded format.

### 5) Train (GPU, ~3 h on A100/H100 for 20k steps)

```bash
wandb login   # one-time

python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

Single-line config overrides via `--override`:

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1 \
    --override data.batch_size=128 \
    --override optim.lr=5e-5 \
    --override optim.max_steps=10000
```

Resume from a checkpoint:

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --resume_from ckpts/m1_llada_variant_c/step_0010000.pt
```

> ⚠️ **After §1.5 + §1.6**: VariantC's state_dict shape changed (RMSNorm
> keys, SwiGLU keys, dropped `pos_emb` / `prefix_proj`, added RoPE
> module). Pre-§1.5 checkpoints WILL NOT load. Start a fresh run and
> only resume from checkpoints saved on or after the refactor.

What to watch in wandb (project `tata-delta-model`, group `M1-llada`):
- `train/loss` decreasing from step 0 onward.
- `train/c_label_mean` rising from ~0.5 baseline toward 0.9+ as the
  delta model gets better.
- `val/mse_by_gap_{1..5}` — gap=1 should be much lower than gap=5.
- `gsm8k/accuracy_delta` — held-out subset every `gsm8k_every` steps;
  target by step 20000 is within 5pp of vanilla baseline.

At startup with `data.preload: true` you should see, before the first
training step:

```
[dataset] preloaded 4500/5000 samples into RAM (~72000 MiB) for split='train'
[dataset] preloaded 500/5000 samples into RAM (~8000 MiB) for split='train'
```

(Both messages say `split='train'` because the train/val partition is
applied via `index_filter`, not split name.) If those lines are missing,
preload is off and you'll hit the disk thrash described above.

Per-step timing breakdown is also printed every `log_every` step:

```
[time] step=  500 data: 18.7ms(11%) fwd: 50.0ms(30%) bwd: 90.0ms(54%) loss:  5.2ms( 3%) h2d:  3.1ms( 2%) opt:  1.8ms( 1%)
```

Sections: `data` (loader/disk), `h2d` (host→device + GPU embed lookup),
`fwd` (variantc forward), `loss` (composite_loss), `bwd` (backward +
grad clip), `opt` (lr update + optimizer step), and `val` when
validation runs. Numbers are CUDA-synced means since the previous
`[time]` line. With preload on, `data` should be small (compute time
dominates). If `data` is still high (>30%), check that preload is
actually on at startup and consider getting `/dev/shm` raised so
workers can come back (see `improvements.md` §2.1).

Halt criteria (from `implementation_plan.md` §12):
- Step 5000: GSM8K subset accuracy ≥ 0.5 × vanilla. Below 0.3 →
  stop and revisit loss weights / model size.
- Step 20000: full GSM8K accuracy ≥ vanilla − 0.05.

### 6) Final eval — sweep conf-threshold

```bash
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --conf_thresholds 0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_v0.json
```

The eval inherits the same `--factor` / `--threshold` flags as collect
(default `--factor 1.0`). **Whatever mode you collected with, eval must
match** — otherwise the delta model is being tested in a regime it
never trained on. The cache's `manifest.json["decoding"]` field
records the collect-time setting; pass the matching flag here.

Pass criteria for the eval smoke run:
- prints `[gsm8k] decoding mode: factor=1.0` (or `threshold=…`) once
  at startup. If this line is missing or wrong, the eval and collect
  modes are out of sync.
- per-`conf_threshold` block returns a dict with non-NaN
  `accuracy_hybrid`, `accuracy_vanilla`, `speedup_ratio`.

Output dict per threshold contains:
- `accuracy_hybrid`, `accuracy_vanilla`, `accuracy_delta`
- `speedup_ratio` (vanilla wall-time / hybrid wall-time)
- `mean_rollbacks` per problem
- `mean_backbone_calls` per problem (8 = no rollbacks at all,
  matches passes-per-block; > 8 = rollbacks happened)

Plot `accuracy_delta` vs `speedup_ratio` to find the working point.

---

## Things to verify on first run (best-effort guesses I made)

These are spots in the code where I wrote the most-likely-correct
thing, but couldn't verify without GPU access. Check on first run.

1. **Nemotron v2 record schema.** `delta_model/data/collect_llada.py:_default_prompt_extractor`
   tries keys `messages` → `input` → `prompt` → `question` → `query`
   → `instruction` and falls back to a clear `KeyError`. If none
   match, edit the function or pass a custom extractor. The smoke
   test (step 3) bypasses this — it's only relevant at step 4.

2. **LLaDA wrapper paths.** In `delta_model/train.py:_load_backbone_components`
   we expect:
   ```
   model.model.transformer.wte      → token embedding
   model.model.transformer.ln_f     → final norm
   model.model.transformer.ff_out   → lm_head (weight-tied to wte)
   ```
   If `LLaDAModelLM` wraps differently in your Fast-dLLM checkout,
   adjust those three lines.

3. **prefix-cache forked loop correctness.** `collect_llada.collect_one_sample`
   ports `generate_with_prefix_cache` and adds hooks. The smoke test
   (step 3) catches gross issues. For deeper verification: diff the
   generated tokens from collect vs. running Fast-dLLM directly on
   the same prompt — should match exactly at the same `--threshold`
   or `--factor`. Note that in `factor` mode, our port of
   `get_transfer_index_dynamic` (`llada_runtime.py:_get_transfer_index_dynamic`)
   uses tensor ops to find the first rank that fails its threshold;
   it preserves the boundary-bump quirk at `top_i ∈ {0, M-1}` so
   committed-token counts match the reference impl.

4. **Cross-attn head shapes.** `delta_model/models/variant_c.py` now
   passes `prefix_kv` K/V directly into cross-attn (no re-projection,
   no re-rotation — see §1.5). This requires `n_kv_heads * d_head == d_model`,
   which holds for LLaDA (no GQA). Dream (GQA factor 8) would need to
   either expand the KV bank or add a `kv_proj` before cross-attn;
   flagged for M3.

5. **RoPE theta surfaced from config.** `_load_backbone_components`
   reads `model.config.rope_theta` and `model.config.layer_norm_eps`
   off the loaded LLaDA and forwards them into VariantC. Look for the
   `[train] backbone hyperparams: {…}` line at startup; if it prints
   defaults (`rope_theta=1e6`, `rms_eps=1e-5`) instead of LLaDA's
   actual values, the wrapper class doesn't expose those attrs and
   needs adjusting.

6. **`block_start_pos` dataset field.** `dataset.py:__getitem__`
   computes `block_start_pos = sample["prompt_len"] + b * BLOCK_LENGTH`.
   This depends on the cache having `prompt_len` recorded per sample
   (which `collect_llada` does). If you're feeding a custom dataset
   without `prompt_len`, you'll need to compute it some other way or
   patch the dataset.

---

## File map (where things live)

```
tata/
  scoping.md                                 — design + locked decisions
  implementation_plan.md                     — file layout, shapes, M2 hooks
  improvements.md                            — perf / correctness / data-collection backlog
  usage.md                                   — this file
  delta_model/                               — the importable package
    llada_runtime.py                         — LLaDA + Fast-dLLM helpers (self-contained)
    data/
      schema.py                              — constants used everywhere
      collect_llada.py                       — cache builder (per-sample .pt files)
      dataset.py                             — TataDeltaDataset (per-sample OR shard mode)
      repack.py                              — converts per-sample .pt → multi-sample shards (§1.4)
      sample_prompts.txt                     — built-in smoke-test prompts
    models/
      heads.py                               — Δh head (zero-init), conf head
      variant_c.py                           — M1 model
    losses.py                                — composite MSE + KL + BCE
    train.py                                 — AdamW + cosine + wandb
    inference/hybrid_runner.py               — generate_with_delta + rollback
    eval/
      shared_mass.py                         — overlap metric
      gsm8k_e2e.py                           — end-to-end eval harness
    sanity/
      test_collect_roundtrip.py              — cache-format check (T3)
      test_zero_init.py                      — model + loss invariant check (T2)
      test_partial_full_forward_equivalence.py  — §3.1 position-id verification (T4)
    configs/m1_llada_variant_c.yaml          — default M1 config
```

## Common error → fix table

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'delta_model'` | wrong cwd | `cd` into the tata repo root, or set `PYTHONPATH=/path/to/tata` |
| `FileNotFoundError: Fast-dLLM v1 not found at …` | Fast-dLLM source tree missing | clone github.com/NVlabs/Fast-dLLM into `external/`, or pass `--fast_dllm_path` |
| `FileNotFoundError: No cached samples found at cache_v1/llada/train` | collect didn't run, or path mismatch | re-run collect, or update `cfg.data.cache_root` |
| `KeyError: 'Could not find a prompt field in record …'` | Nemotron schema mismatch | adjust `_default_prompt_extractor` in `collect_llada.py` |
| `RuntimeError: No LLaDA transformer blocks found` | model wrapping changed | update class names in `_find_last_block` (collect + hybrid_runner) |
| All samples skipped: "prompt too short (< 32 tokens)" | extracted prompts all very short | swap to a different `--prompts_file` or fix the extractor |
| wandb hangs at init | not logged in | `wandb login` once, then re-run |
| 401/403 from HuggingFace on Nemotron load | not logged in or Nemotron not accepted | `huggingface-cli login` and accept Nemotron license at the dataset's HF page |
| OOM during training | batch too big for the GPU | `--override data.batch_size=128` (or 64) |
| OOM during collect | LLaDA + cache snapshots peak | reduce concurrent samples / restart, or rebuild with a more-aggressive decoding setting (`--factor 1.5` or `--threshold 0.85`) so blocks finish in fewer passes |
| Eval and collect disagree on `n_passes_actual` distribution | train/eval decoding-mode mismatch | check `manifest.json` → `decoding`. Cache must be rebuilt with the same mode (`factor` or `threshold`) that eval will use |
| GSM8K mid-train eval fails | usually a rollback path bug under bf16 dtype mismatch | logged + skipped (training continues), check the printed traceback in the run log |
| `RuntimeError: Error(s) in loading state_dict for VariantC: Missing key(s) in state_dict: ".../weight"` | resuming a pre-§1.5 checkpoint into the new architecture | start fresh; the model definition changed (RMSNorm + SwiGLU + RoPE) and old shapes don't fit |
| `KeyError: 'block_start_pos'` in train loop | dataset built before §1.5 (cached `__pycache__`) | clear `delta_model/data/__pycache__/` and re-run |
| `[time]` shows `data:>50%` consistently | per-sample I/O is the bottleneck | run §4b shard repack, and/or get `/dev/shm` bumped so workers can prefetch (see `improvements.md` §2.1) |
| Bus error in DataLoader after running fine for many steps | cluster `/dev/shm` is too small for any worker IPC at all | already fixed by `num_workers: 0` in the default config; if you flipped it back without a shm bump, revert |

## Cleanup / partial-data scenarios

Cache is fully resumable: collect skips files that already exist
(`if out_path.exists(): continue`). Safe to Ctrl-C and re-run.

To rebuild the test set from scratch (e.g. after changing
`subset_ratios` for the test partition), delete:

```bash
rm cache_v1/llada/manifest.json
rm -rf cache_v1/llada/test
```

The hash partition will produce the same test IDs for any given
subset ratios, so you can also just rebuild the manifest without
re-collecting if the per-subset ratios for test stayed constant.
