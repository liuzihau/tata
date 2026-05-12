# tata — operator's manual

Run this when you're about to launch something on a GPU box. For *why*
any of this exists, see `design.md`. For *how* the internals work
(contracts, code structure, status of every component), see
`engineering.md`.

`tata/` is **a standalone repo** — no cross-package imports from
sibling `peft_project/` trees. The only external code dependency is
**Fast-dLLM v1**, found at runtime via path / env var (§ Prereqs).

---

## Quickstart — post-M1.5: T4 sweep + M1.6 + T5 collect, all in parallel

After M1.5 (val KL plateaued ≈ trial-2, GSM8K 0.40 within 1σ of 0.44),
three concurrent next actions. The first runs on the existing M1.5
checkpoint; the second on the existing cache; the third writes to a
new cache dir, so all three can be launched at the same time without
racing.

### Action 1 — T4 sweep on M1.5 checkpoint (free, ~30 min)

```bash
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_5_tier1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.60,0.70,0.75,0.80,0.85,0.90,0.95 \
    --out_json eval_results/t4_m1_5_thresh_sweep.json
```

The 0.85 default was design-time, not tuned. GSM8K is non-monotone
in this threshold. If a different threshold lands at 0.46+, the M1.5
result was undersold by a bad operating point. Note the winner — pass
it to subsequent training via `--override log.gsm8k_per_pos_threshold=<w>`.

### Action 2 — M1.6 (T6 `λ_mse = 0`) on the existing cache (~3–4 h)

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_6_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

Lands in `ckpts/m1_6_lambdamse0_llada_variant_c/`, wandb group
`M1.6-lambdamse0-llada`. Reads the **same** `cache_v1/llada/` as
M1.5 — directly comparable to M1.5's val KL trajectory. Watch:

- `train/mse` and `val/mse` will be reported (mse_space still logs it
  for comparability) but contribute zero to the loss. The interesting
  signal is `train/kl` and `val/kl`.
- If M1.6's val KL ends ≤ M1.5's (≤ 0.293), MSE was redundant and we
  ship the 2-term loss for cleanliness.
- If higher, MSE was a low-frequency anchor; keep it but maybe tune
  λ down (try 0.3 next).

### Action 3 — T5 collect (scale data 5k → 20k) (~12–20 h)

```bash
# Lands in a NEW cache dir so it doesn't race with M1.6 on
# cache_v1/llada/manifest.json.
python -m delta_model.data.collect_llada \
    --n_train 20000 --n_test 800 \
    --output_root cache_v1_20k/llada \
    --fast_dllm_path external/Fast-dLLM/v1
python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1_20k/llada/test/sample_*.pt" | tail -5     # T3 schema check
```

The hash-stable test split means the 800 test prompts in
`cache_v1_20k/llada/test/` are **identical** to `cache_v1/llada/test/`
(same hashes, same `shuffle_seed=42`). Direct A/B vs trial-2/M1.5 on
the same held-out test prompts.

Storage: ~480 GB at n_train=20000 + n_test=800 with prefix_window=64.
If tight, drop to n_train=15000 (~360 GB).

Once T5 collect lands, the T5 training config is one copy of
`m1_5_llada_variant_c.yaml` with `data.cache_root: cache_v1_20k/llada`
and fresh `run_name` / `log.group` / `checkpoint.out_dir`. Not creating
that config yet — wait for collect to finish so we can pick batch /
epoch knobs against the actual sample count.

### Decision tree after these three land

| signal | interpretation | next move |
|---|---|---|
| T4 winner ≥ 0.46 | Bad operating point for M1.5 | Adopt as default; M1.5 wasn't underperforming |
| M1.6 val KL ≤ M1.5 | MSE was redundant | Ship 2-term loss; carry into T5 |
| M1.6 val KL > M1.5 | MSE was a soft anchor | Keep MSE but tune λ ∈ {0.3, 0.5} |
| T5 val KL < 0.20 | Data was the binding constraint | Continue scaling; T7 capacity bump if val plateau |
| T5 val KL ≈ M1.5 | Data isn't the binding constraint | Look at capacity (T7) or trajectory drift (T8) |

---

## Quickstart — Tier 1 (M1.5; reuses trial-2 cache, no recollect)

If you have a trial-2 cache at `cache_v1/llada/` and a trial-2
checkpoint at `ckpts/m1_llada_variant_c/`, this is the full Tier 1
sequence. The Tier-1 trial uses a **separate config**
(`m1_5_llada_variant_c.yaml`), wandb group (`M1.5-tier1-llada`), and
checkpoint dir (`ckpts/m1_5_tier1_llada_variant_c/`) — the trial-2
artifacts stay untouched as the baseline of record.

Total ~3–4 h wall time + ~30 min for T4.

```bash
# 0. Pre-flight compile + sanity
python3 -m py_compile \
    delta_model/{losses,train}.py \
    delta_model/data/dataset.py \
    delta_model/models/variant_c.py \
  && echo OK
python -m delta_model.sanity.test_zero_init                          # raw-MSE invariant still holds (test uses default kwarg)

# 1. (free) T4 — per_pos_threshold sweep on the EXISTING trial-2 checkpoint.
#    No retrain. Could move 0.44 → ~0.50 at zero cost.
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.75,0.80,0.85,0.90,0.95 \
    --out_json eval_results/t4_trial2_thresh_sweep.json
# Pick the best per_pos_threshold — that's your new gsm8k_per_pos_threshold default.

# 2. Launch the Tier-1 trial. Lands in a SEPARATE ckpt dir — the trial-2
#    checkpoint stays in place untouched.
python -m delta_model.train \
    --config delta_model/configs/m1_5_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1 \
    --override log.gsm8k_per_pos_threshold=<T4 winner>     # optional

# 3. Final Tier-1 eval (same threshold sweep, fresh checkpoint).
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_5_tier1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_5_tier1.json
```

What to watch in wandb during Tier-1 training (wandb group
`M1.5-tier1-llada`, run name `m1_5_tier1_llada_variant_c_seed42`):

- `train/mse` should now sit at O(1) instead of O(10), since MSE is
  computed in `final_norm` space (T1). If you see it at O(10), the
  config's `loss.mse_space` wasn't picked up.
- `train/kl` should drop faster than trial 2 — same KL term, but now
  MSE is pulling in the same direction instead of the orthogonal
  raw-h direction.
- `val/kl` ÷ `train/kl` should narrow toward 1× (was ~4× at trial 2).
  If it stays at 4×, T1 didn't help; revisit T3 dropout strength.
- `val/mse_by_gap_1` should drop fastest under T2 (i_ref=0 → gap=1
  pairs are the most over-sampled bin).
- At startup the train log should print a line like
  `[train] i_ref-biased sampler: ref0_weight_multiplier=3.0 → P(i_ref=0) ≈ 0.60 (...)`.
  If it doesn't appear, T2 didn't fire.

### A/B against trial-2

To re-launch trial-2 exactly (e.g. to confirm baseline reproducibility),
use the original config:

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

That lands in `ckpts/m1_llada_variant_c/` and wandb group `M1-llada` —
same place the trial-2 result lives.

---

## Quickstart — trial 2 (recollect + retrain)

If you just landed the §3.1/§3.2 changes from `engineering.md` and want
to retrain on a fresh cache, this is the full sequence. Total ~5–8 h
wall time.

```bash
# 1. Pre-flight: compile + sanity checks (~1 min)
rm -rf delta_model/{data,models,sanity,inference,eval}/__pycache__
python3 -m py_compile \
    delta_model/data/{schema,collect_llada,dataset,repack}.py \
    delta_model/models/{heads,variant_c}.py \
    delta_model/{losses,train}.py \
    delta_model/inference/hybrid_runner.py \
    delta_model/eval/{shared_mass,gsm8k_e2e}.py \
    delta_model/sanity/{test_zero_init,test_collect_roundtrip,test_partial_full_forward_equivalence,inspect_llada_modeling}.py \
  && echo OK
python -m delta_model.sanity.test_zero_init                                                      # T2
python -m delta_model.sanity.test_partial_full_forward_equivalence --fast_dllm_path external/Fast-dLLM/v1   # T4 (expect ✓ EXPECTED)

# 2. Move stale artifacts aside (don't delete; useful as a reference)
mv cache_v1/llada                cache_v1/llada_pre_3_1                       2>/dev/null || true
mv ckpts/m1_llada_variant_c      ckpts/m1_llada_variant_c_pre_3_2             2>/dev/null || true

# 3. (Optional) 5-prompt smoke recollect (~3 min, no HF login)
python -m delta_model.data.collect_llada \
    --prompts_file delta_model/data/sample_prompts.txt \
    --output_root cache_v1/llada_smoke \
    --fast_dllm_path external/Fast-dLLM/v1
python -m delta_model.sanity.test_collect_roundtrip "cache_v1/llada_smoke/test/sample_*.pt"      # T3

# 4. Real recollect (~4–7 h)
python -m delta_model.data.collect_llada \
    --n_train 5000 --n_test 800 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1
python -m delta_model.sanity.test_collect_roundtrip "cache_v1/llada/test/sample_*.pt" | tail -5  # T3 again

# 5. Train (~3–4 h for 20k steps)
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1

# 6. Final eval (sweeps the per-position threshold)
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_trial2.json
```

What to watch in wandb during training:
- `train/loss` should drop from the start.
- `val/loss` should track `train/loss`; if they diverge sharply at e.g. 8k, the model is overfitting.
- `gsm8k/accuracy_hybrid` at intermediate checkpoints — should **trend up** with training. If it declines monotonically like in trial 1, §3.2 didn't help and we move to model-side investigation (see `engineering.md` §11 open questions).
- `[time]` line printed every `log_every` step: with `data.preload: true` (default), `data` should be small; `fwd / bwd` dominate.

If something breaks, see `## Common errors` at the bottom.

---

## cwd and Python path

All commands assume cwd = the tata repo root. After cloning:

```
$ cd <wherever>/tata && ls
delta_model/  design.md  engineering.md  usage.md  ...
```

Imports are relative within `delta_model/`. `python -m delta_model.X`
works because `delta_model/` is in the cwd. From elsewhere, use
`PYTHONPATH=/abs/path/to/tata python -m delta_model.X`.

---

## Prerequisites

| step | GPU? | HF login? | Fast-dLLM? | Time |
|---|---|---|---|---|
| T1 compile-check | no | no | no | <5 s |
| T2 zero-init | no | no | no | ~5 s |
| T3 cache-format roundtrip | no | no | no | ~10 s per cache |
| T4 partial-vs-full forward | **yes** | no | **yes** | ~30 s |
| Smoke recollect (5 prompts) | **yes** | no | **yes** | ~3 min |
| Real recollect (5800 prompts) | **yes** | **yes** | **yes** | ~4–7 h |
| Train (20k steps) | **yes** | no | **yes**¹ | ~3–4 h |
| Final eval | **yes** | no | **yes** | depends on N |

¹ training needs Fast-dLLM v1 because we lift `final_norm`, `lm_head`,
and `token_embed` off the loaded LLaDA backbone.

### Fast-dLLM v1 placement

Recommended layout (cwd = the tata repo root):

```
tata/
├── delta_model/
│   ├── data/sample_prompts.txt
│   └── ...
└── external/Fast-dLLM/v1/            ← clone of github.com/NVlabs/Fast-dLLM
    └── llada/model/modeling_llada.py
```

```bash
mkdir -p external && cd external
git clone https://github.com/NVlabs/Fast-dLLM
cd ..
```

Discovery order at runtime:
1. CLI flag: `--fast_dllm_path external/Fast-dLLM/v1`
2. Env var: `export FAST_DLLM_V1_PATH=/abs/path/to/Fast-dLLM/v1`
3. Default: `external/Fast-dLLM/v1` (relative to cwd)

### Python deps

```bash
pip install "torch>=2.4" transformers>=4.44 datasets h5py wandb pyyaml \
            numpy huggingface_hub
```

LLaDA-8B is downloaded from `GSAI-ML/LLaDA-8B-Instruct` on first load
(public, no auth). HF login is only required for the gated Nemotron
Post-Training v2 dataset used in real collect.

### One-time auth

```bash
wandb login                 # before training
huggingface-cli login       # before real collect (Nemotron is gated)
```

---

## Diagnostic tests reference

| # | what it tests | command |
|---|---|---|
| T1 | Syntax / imports across the package | `python3 -m py_compile delta_model/...` (see quickstart) |
| T2 | DeltaHead zero-init invariant + full model forward + per-position loss path | `python -m delta_model.sanity.test_zero_init` |
| T3 | Cache schema matches `data/schema.py`; field shapes/dtypes; optional `prefix_kv_pad_mask` if present | `python -m delta_model.sanity.test_collect_roundtrip "cache/.../sample_*.pt"` |
| T4 | Documents that Fast-dLLM's partial-forward path diverges from a hypothetical full forward at the same `x` state (expected behavior on this LLaDA build) | `python -m delta_model.sanity.test_partial_full_forward_equivalence --fast_dllm_path external/Fast-dLLM/v1` |
| diag | Print loaded LLaDA's forward signatures + `RotaryEmbedding.forward` source + attention method source + targeted grep hits | `python -m delta_model.sanity.inspect_llada_modeling --fast_dllm_path external/Fast-dLLM/v1` |

Pass criteria:
- **T1**: prints `OK`.
- **T2**: prints `delta_h.abs().max() = 0.000000e+00` and `✓ zero-init sanity passed`.
- **T3**: prints `N OK, 0 bad.` for each sample.
- **T4**: prints `✓ EXPECTED — both partial-forward paths diverge from the full forward`. (Both partial paths *should* diverge; this is the documented Fast-dLLM behavior. If T4 instead prints `⚠ NOTABLE`, the modeling code has been updated to make them agree — informational.)

T4 details and the alignment rationale: `engineering.md` §6.

---

## Real collect (Nemotron, gated)

```bash
huggingface-cli login    # once; accept the Nemotron license at the dataset page

python -m delta_model.data.collect_llada \
    --n_train 5000 --n_test 800 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Hash-stable test split: re-running with a larger `--n_train` will not
touch the same 800 test samples, so you can grow the train set across
runs without contaminating eval.

Knobs (see `engineering.md` §1.2 for the full list):
- `--subset_ratios '{"chat":0.4,"math":0.4,"code":0.2,"stem":0.0}'` — change the mix.
- `--factor 1.0` (default) — Fast-dLLM dynamic per-rank threshold. Paper-recommended; matches the baseline at eval.
- `--threshold 0.9` — fixed-threshold mode (mutually exclusive with `--factor`).
- `--prefix_window N` (default 64) — number of last-prefix tokens cached per block. Must be ≥ 32. The cache also stores a pad mask so prompts shorter than `prefix_window` are kept, not skipped.
- `--start_index N` — resume a partially-done run.
- `--limit N` — testing cap (overrides `n_train + n_test`).

Storage: ~120 GB at default 5800 samples (5000 train + 800 test) with
`prefix_window=64`. Cache files are resumable: if interrupted, just
re-run — existing files are skipped.

---

## Training

```bash
wandb login   # once

python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

Single-line config overrides via `--override key=value` (dot notation):

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

> Pre-§1.5 / §3.2 checkpoints **will not load** — state_dict keys changed
> (RoPE / RMSNorm / SwiGLU / per-position conf head). Start fresh.

At startup you should see (with `data.preload: true`, the default):

```
[dataset] preloaded ~4500/5000 samples into RAM (~72000 MiB) for split='train'
[dataset] preloaded ~500/5000  samples into RAM (~8000 MiB) for split='train'
[train] backbone hyperparams: {'rope_theta': 1e6, 'rms_eps': 1e-5, ...}
```

Per-step timing (every `log_every` steps):

```
[time] step=  500 data: 18.7ms(11%) fwd: 50.0ms(30%) bwd: 90.0ms(54%) loss:  5.2ms(3%) h2d: 3.1ms(2%) opt: 1.8ms(1%)
```

With preload on, `data` should be small (compute-bound). If `data > 30%`,
check that preload is actually on at startup, and see `engineering.md`
§1.4 (open items) for follow-ups.

What to watch in wandb (project `tata-delta-model`, group `M1-llada`):
- `train/loss` decreasing from step 0.
- `train/c_label_mean` rising from ~0.5 baseline toward 0.9+ as the
  delta model gets better.
- `val/mse_by_gap_{1..5}` — gap=1 should be much lower than gap=5.
- `gsm8k/accuracy_hybrid` mid-training — trend should be **rising**.

Halt criteria (rough guidance, not a hard rule):
- Step 5000: GSM8K subset accuracy ≥ 0.5 × vanilla. Below 0.3 → stop and revisit loss weights / model size.
- Step 20000: full GSM8K accuracy ≥ vanilla − 0.05.

---

## Final eval

```bash
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_v0.json
```

`--per_pos_thresholds` controls the §3.2 agreement-decode gate. Lower
= more lenient (more delta commits, faster, riskier). Higher = more
conservative (more rollbacks, slower, safer).

**Decoding mode must match collect.** Pass `--factor 1.0` (default) or
`--threshold 0.9` — whichever was used to build the cache. The
cache's `manifest.json["decoding"]` records this.

Output per threshold:
- `accuracy_hybrid`, `accuracy_vanilla`, `accuracy_delta`
- `speedup_ratio` (vanilla wall-time / hybrid wall-time)
- `mean_rollbacks`, `mean_backbone_calls`, `mean_disagreements` per problem

Plot `accuracy_delta` vs `speedup_ratio` to find the working point.

---

## Common errors

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'delta_model'` | wrong cwd | `cd` into tata repo root, or `PYTHONPATH=/path/to/tata` |
| `FileNotFoundError: Fast-dLLM v1 not found at …` | Fast-dLLM not cloned | clone NVlabs/Fast-dLLM into `external/`, or pass `--fast_dllm_path` |
| `FileNotFoundError: No cached samples found at cache_v1/llada/train` | collect didn't run | run real collect; or update `cfg.data.cache_root` |
| `KeyError: 'Could not find a prompt field in record …'` | Nemotron record schema mismatch | adjust `_default_prompt_extractor` in `collect_llada.py` |
| `RuntimeError: No LLaDA transformer blocks found` | wrapper class renamed in your Fast-dLLM checkout | update class names in `_find_last_block` (collect + hybrid_runner) |
| `All samples skipped: "prompt too short (< 1 tokens)"` | empty prompts in your input file | filter the prompts file; pad-and-mask handles short-but-nonempty prompts but cannot handle empty |
| wandb hangs at init | not logged in | `wandb login` once, then re-run |
| 401/403 from HuggingFace on Nemotron | gated dataset; not accepted | `huggingface-cli login` and accept the Nemotron license on its HF page |
| OOM during training | batch / preload / RAM | `--override data.batch_size=128`; or `--override data.preload=false` if RAM-constrained |
| `RuntimeError: Error(s) in loading state_dict for VariantC: Missing key(s) ".../weight"` | resuming a pre-§3.2 checkpoint into the new architecture | start fresh; model definition changed (RoPE + RMSNorm + SwiGLU + per-pos conf) |
| `KeyError: 'block_start_pos'` or `'prefix_kv_pad_mask'` in train loop | stale `__pycache__` | `rm -rf delta_model/data/__pycache__` and re-run |
| `[time]` shows `data: >30%` after preload | preload didn't fire | check startup for `[dataset] preloaded N/M samples` line; if missing, look at `data.preload` in config |
| Bus error in DataLoader (SIGBUS) | cluster `/dev/shm` too small for any worker IPC | default config already uses `num_workers: 0`; if you flipped it back, revert (see `engineering.md` §10) |
| Eval and collect disagree on `n_passes_actual` distribution | decoding-mode mismatch | check `manifest.json["decoding"]`; pass the matching `--factor` or `--threshold` at eval |
| GSM8K mid-train eval fails | usually a dtype mismatch in the rollback path | logged + skipped (training continues); see the printed traceback |

---

## Cleanup / partial-data scenarios

Cache is fully resumable: collect skips files that already exist. Safe
to Ctrl-C and re-run.

To rebuild the test set from scratch (e.g. after changing
`subset_ratios` for the test partition), delete:

```bash
rm cache_v1/llada/manifest.json
rm -rf cache_v1/llada/test
```

The hash partition produces the same test IDs for any given subset
ratios, so you can rebuild the manifest without re-collecting if
`subset_ratios` for test stayed constant.

If you want to keep a "broken cache" around as a reference while
working on a new one, move-aside is safer than delete:

```bash
mv cache_v1/llada  cache_v1/llada_pre_3_1
```

---

## What lives where

```
design.md         — research framing: problem, idea, prior work, milestones
engineering.md    — code-level contracts, current status of every component
usage.md          — this file: commands to run, error handling
README.md         — short pointer to the three above (if present)
```

Old docs (`scoping.md`, `implementation_plan.md`, `improvements.md`) are
kept as stubs pointing here / to `engineering.md`; their content has been
consolidated.
