# tata — operator's manual

Run this when you're about to launch something on a GPU box. For *why*
any of this exists, see `design.md`. For *how* the internals work
(contracts, code structure, status of every component), see
`engineering.md`.

`tata/` is **a standalone repo** — no cross-package imports from
sibling `peft_project/` trees. The only external code dependency is
**Fast-dLLM v1**, found at runtime via path / env var (see Prereqs).

The current end-to-end pipeline:

```
collect_llada  →  thin_cache  →  repack --rm_source  →  sanity  →  train  →  gsm8k_e2e  →  plot_sweep
                  (optional)                            (preflight)            (sweep)      / plot_metrics
```

A clean run from scratch is in [Quickstart](#quickstart). Sections
below cover each step in detail.

---

## Quickstart

A full run from no cache to a per-threshold sweep plot. Total ~5–7 h
on a single H100-class node (collect dominates).

```bash
# 0. Pre-flight (no GPU / cache needed) — ~30 s
python3 -m py_compile \
    delta_model/{train,inference/hybrid_runner}.py \
    delta_model/data/{schema,collect_llada,dataset,repack,thin_cache}.py \
    delta_model/eval/{gsm8k_e2e,plot_metrics,plot_sweep}.py \
  && echo OK
python -m delta_model.sanity.test_zero_init
python -m delta_model.sanity.test_interleaved_sampler

# 1. Collect (~4–7 h depending on n_train) — needs HF + Fast-dLLM
huggingface-cli login                                                     # once
python -m delta_model.data.collect_llada \
    --n_train 20000 --n_test 800 \
    --output_root cache_v1_20k/llada \
    --fast_dllm_path external/Fast-dLLM/v1

# 2. Thin the train split in-place (drop block 7, cap iters to 5,
#    shrink KV window 64 → 32). Test split is left full for fair eval.
python -m delta_model.data.thin_cache --cache_root cache_v1_20k/llada
python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1_20k/llada/train/sample_*.pt" | tail -3                       # T3 on thinned train
python -m delta_model.sanity.test_collect_roundtrip \
    "cache_v1_20k/llada/test/sample_*.pt"  | tail -3                       # T3 on full-fidelity test

# 3. Repack into shards, deleting per-sample files as each shard lands
python -m delta_model.data.repack --cache_root cache_v1_20k/llada \
    --shard_size 50 --output_subdir shards --rm_source

# 4. Train — current best config (~3–4 h for 20k steps)
wandb login                                                               # once
python -m delta_model.train \
    --config delta_model/configs/m1_5_data20k_interleaved_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1

# 5. GSM8K per_pos_threshold sweep on the best-by-GSM8K checkpoint
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt $(ls ckpts/m1_5_data20k_interleaved_llada_variant_c/best_gsm8k_step*.pt) \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_5_data20k_interleaved_sweep.json

# 6. Visualize the sweep
python -m delta_model.eval.plot_sweep \
    eval_results/m1_5_data20k_interleaved_sweep.json \
    --out plots/m1_5_data20k_interleaved_sweep.png
```

Multi-run comparison of training curves at the end:

```bash
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_data20k_llada_variant_c/metrics.jsonl \
    ckpts/m1_5_data20k_preload_llada_variant_c/metrics.jsonl \
    ckpts/m1_5_data20k_interleaved_llada_variant_c/metrics.jsonl \
    --metric train/kl gsm8k/accuracy_hybrid gsm8k/speedup_ratio throughput/samples_per_sec \
    --smooth 25 \
    --out plots/sampler_3way_core.png
```

---

## Prerequisites

### Python deps

```bash
pip install "torch>=2.4" transformers>=4.44 datasets h5py wandb pyyaml \
            numpy huggingface_hub tqdm matplotlib
```

LLaDA-8B is downloaded from `GSAI-ML/LLaDA-8B-Instruct` on first load
(public, no auth). HF login is only required for the gated Nemotron
Post-Training v2 dataset used in real collect.

### Fast-dLLM v1 placement

Recommended layout (cwd = the tata repo root):

```
tata/
├── delta_model/
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

### One-time auth

```bash
wandb login                 # before training
huggingface-cli login       # before real collect (Nemotron is gated)
```

### cwd and Python path

All commands assume cwd = the tata repo root. Imports are relative
within `delta_model/`; `python -m delta_model.X` works because
`delta_model/` is in the cwd. From elsewhere, use
`PYTHONPATH=/abs/path/to/tata python -m delta_model.X`.

---

## 1 · Collect (`data/collect_llada.py`)

Builds the per-sample cache from Nemotron prompts by running LLaDA +
Fast-dLLM in prefix-cache mode.

```bash
python -m delta_model.data.collect_llada \
    --n_train 20000 --n_test 800 \
    --output_root cache_v1_20k/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Knobs (see `engineering.md` §1.2 for the full set):

- `--subset_ratios '{"chat":0.35,"math":0.35,"code":0.20,"stem":0.10}'` — change the mix.
- `--factor 1.0` (default) — Fast-dLLM dynamic per-rank threshold. Paper-recommended; matches the baseline at eval.
- `--threshold 0.9` — fixed-threshold mode (mutually exclusive with `--factor`).
- `--prefix_window N` (default 64) — last-prefix KV slots cached per block. Must be ≥ 32.
- `--start_index N` — resume a partially-done run.
- `--limit N` — testing cap.

**Hash-stable test split**: re-running with a larger `--n_train` will
not touch the same 800 test samples, so you can grow the train set
across runs without contaminating eval.

**Storage**: ~120 GB at default 5800 samples; ~480 GB at 20k samples
with `--prefix_window 64`. If tight, drop to `--prefix_window 32` and
plan to thin (next step) before sharding.

**Resume**: collect skips files that already exist. Safe to Ctrl-C
and re-run.

---

## 2 · Thin (`data/thin_cache.py`)

In-place cache thinner — drops trailing blocks, caps iterations, and
shrinks the prefix-KV window. Per-file atomic, resumable (skips
already-thinned files). Defaults thin **train only**; test stays at
full fidelity so the held-out eval set is identical run-to-run.

```bash
python -m delta_model.data.thin_cache \
    --cache_root cache_v1_20k/llada \
    --keep_blocks 7 --keep_iters 5 --prefix_window 32
```

Defaults match the current best practice (drop block 7, cap iters at 5,
window 32). Pass `--splits train,test` only if you also want to thin
the test split (you probably don't — it's the eval anchor).

The dataset and `test_collect_roundtrip` (T3) read each sample's
`meta` for `num_blocks`/`max_iter`/`prefix_window`, so thinned-train +
full-test coexist in one cache transparently.

---

## 3 · Repack (`data/repack.py`)

Packs per-sample files into multi-sample shards. With `--rm_source`
each per-sample file is unlinked once the shard containing it is
durably written, so peak disk stays at ~(cache + one shard) instead
of doubling.

```bash
python -m delta_model.data.repack \
    --cache_root cache_v1_20k/llada \
    --shard_size 50 \
    --output_subdir shards \
    --rm_source
```

After repack:
- `cache_v1_20k/llada/shards/{train,test}/shard_NNNNN.pt` — shard files
- `cache_v1_20k/llada/shards_manifest.json` — auto-detected by the dataset

To revert to per-sample mode (assuming you ran *without* `--rm_source`):
delete the `shards/` directory + `shards_manifest.json`.

---

## 4 · Sanity tests (`sanity/`)

| # | command | what it checks | when to run |
|---|---|---|---|
| T1 | `python3 -m py_compile delta_model/...` | syntax / imports | before every sit-and-wait command |
| T2 | `python -m delta_model.sanity.test_zero_init` | DeltaHead zero-init invariant; full forward signature | after model or loss changes |
| T3 | `python -m delta_model.sanity.test_collect_roundtrip "cache/.../sample_*.pt"` | cache schema matches `schema.py` (validates against meta-recorded num_blocks/max_iter/prefix_window) | after any collect or thin run |
| T4 | `python -m delta_model.sanity.test_partial_full_forward_equivalence --fast_dllm_path external/Fast-dLLM/v1` | documents Fast-dLLM partial-forward behavior; prints `✓ EXPECTED` on this LLaDA build | informational; after any LLaDA upgrade |
| T5 | `python -m delta_model.sanity.test_interleaved_sampler` | `InterleavedShardSampler` count / interleaving / coverage / T2 weighting | after sampler changes; no GPU/cache needed |

Pass criteria:
- **T1**: prints `OK`.
- **T2**: prints `delta_h.abs().max() = 0.000000e+00` and `✓ zero-init sanity passed`.
- **T3**: prints `N OK, 0 bad.` for each sample.
- **T4**: prints `✓ EXPECTED — both partial-forward paths diverge from the full forward`.
- **T5**: prints `✓ InterleavedShardSampler sanity passed`.

---

## 5 · Train (`train.py`)

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_5_data20k_interleaved_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

Single-line config overrides via `--override key=value` (dot notation):

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_5_data20k_interleaved_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1 \
    --override data.active_shards=16 \
    --override optim.lr=5e-5
```

Resume from a checkpoint:

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_5_data20k_interleaved_llada_variant_c.yaml \
    --resume_from ckpts/m1_5_data20k_interleaved_llada_variant_c/step_0010000.pt
```

### What to watch at startup

The first ~minute of output tells you the new code paths engaged:

```
[train] val_split='test' → val from 'test' split. train pairs=…  val pairs=…
[train] InterleavedShardSampler: <N> shards, active_shards=8, chunk_size=… (≈… batches/window). ref0_mult=3.0 …
[time] step=    50 data: …ms(<5%) fwd: …ms(30%) bwd: …ms(54%) …
```

🚩 If you instead see `BlockShardSampler (legacy)` or `i_ref-biased
sampler:` for a 20k-cache run, the dataset didn't land in shard mode
(missing `shards_manifest.json` — re-run repack) or the config has
`shard_sampler: block` pinned (legacy reproducibility configs).

### What to watch during training

`<ckpt_dir>/metrics.jsonl` mirrors every `wandb.log` payload — plottable
locally without wandb cloud (`plot_metrics.py`, §7).

- `train/kl` decreasing smoothly. Heavy noise → loader / sampler is wrong.
- `val/kl` decreasing too, no big gap from train (with M1.5 recipe + 20k
  data + InterleavedShardSampler, expect convergence near val/kl 0.28).
- `gsm8k/accuracy_hybrid` mid-training — trend should be **rising**.
- `throughput/samples_per_sec` — interleaved sits a touch below preload.

### Best-checkpoint files (auto-saved)

Every run saves three kinds of checkpoint to `cfg.checkpoint.out_dir`:

| pattern | when | survives `keep_last` |
|---|---|---|
| `step_<NNNNNNN>.pt` | every `cfg.checkpoint.every` steps | no (rotated) |
| `best_val_kl_step<N>.pt` | when `val/kl` hits a new minimum | yes |
| `best_gsm8k_step<N>.pt` | when `gsm8k/accuracy_hybrid` hits a new max | yes |

Use the `best_*` files for final eval. `<ckpt_dir>/best_metrics.json` is
a sidecar that records the running best per metric (persists across
resumes).

---

## 6 · GSM8K eval & sweep (`eval/gsm8k_e2e.py`)

Sweeps `per_pos_threshold` against the same GSM8K slice, comparing
hybrid vs vanilla `generate_with_prefix_cache`.

```bash
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt $(ls ckpts/m1_5_data20k_interleaved_llada_variant_c/best_gsm8k_step*.pt) \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_5_data20k_interleaved_sweep.json
```

Live progress bar shows running per-problem decode times, speedup, and
accuracy — a broken threshold is visible after ~10 problems:

```
gsm8k thr=0.85:  47%|████▋     | 94/200 [03:12<03:36, hyb=1.83s, van=2.05s, x=1.12, acc_h=0.41, acc_v=0.75]
```

Output (one dict per threshold): `accuracy_hybrid`, `accuracy_vanilla`,
`accuracy_delta`, `speedup_ratio`, `mean_rollbacks`,
`mean_backbone_calls`, `mean_revealed_per_block`, `mean_blocks_finished`.

**Decoding mode must match collect.** Pass `--factor 1.0` (default) or
`--threshold 0.9` — whichever was used to build the cache. The cache's
`manifest.json["decoding"]` records this.

For final eval against both kinds of "best" checkpoint:

```bash
for best in best_gsm8k_step best_val_kl_step; do
    python -m delta_model.eval.gsm8k_e2e \
        --delta_ckpt $(ls ckpts/<run>/${best}*.pt) \
        --fast_dllm_path external/Fast-dLLM/v1 \
        --n_problems 200 \
        --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
        --out_json eval_results/<run>_${best}_sweep.json
done
```

---

## 7 · Plot

### Threshold-sweep visualizer (`eval/plot_sweep.py`)

Three panels: accuracy–speed Pareto with the "win zone" shaded;
accuracy bars (left axis) + speedup line (right axis) vs threshold;
rollback bars vs threshold. Single file = polished single-run plot;
multiple files = overlay (grouped bars, per-file colour).

```bash
# Single sweep
python -m delta_model.eval.plot_sweep \
    eval_results/m1_5_data20k_interleaved_sweep.json \
    --out plots/m1_5_data20k_interleaved_sweep.png

# Overlay (e.g. preload vs interleaved, post-fix)
python -m delta_model.eval.plot_sweep \
    eval_results/m1_5_data20k_preload_sweep.json \
    eval_results/m1_5_data20k_interleaved_sweep.json \
    --out plots/preload_vs_interleaved_sweep.png
```

Prints a one-line summary per file: `best acc 0.775 @ thr=0.85 (speedup
0.37x) | 1 threshold(s) faster than vanilla`.

### Training-metrics overlay (`eval/plot_metrics.py`)

Multi-run comparison of `<ckpt_dir>/metrics.jsonl` series. Smoothing
window auto-skips sparse metrics (e.g. `gsm8k/*` logged every 2000
steps).

```bash
# Single run, popup window
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_data20k_interleaved_llada_variant_c/metrics.jsonl

# Multi-run comparison on specific metrics
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_data20k_llada_variant_c/metrics.jsonl \
    ckpts/m1_5_data20k_preload_llada_variant_c/metrics.jsonl \
    ckpts/m1_5_data20k_interleaved_llada_variant_c/metrics.jsonl \
    --metric train/kl gsm8k/accuracy_hybrid gsm8k/speedup_ratio throughput/samples_per_sec \
    --smooth 25 \
    --out plots/sampler_3way_core.png

# Per-bin val diagnostics (regex)
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_data20k_interleaved_llada_variant_c/metrics.jsonl \
    --metric "val/mse_by_gap_.*" \
    --out plots/val_by_gap.png
```

Notes:
- Run label = checkpoint folder name (`ckpts/<name>/metrics.jsonl`
  → `<name>`).
- `--metric` accepts exact names or regex (`train/.*`,
  `val/mse_by_gap_[0-9]+`). Default = metrics common across the runs.
- Trial-2 / M1.5-5k ran *before* the JSONL mirror landed; their
  history lives only in wandb. M1.6 onwards have `metrics.jsonl`.

---

## Common errors

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'delta_model'` | wrong cwd | `cd` into tata repo root, or `PYTHONPATH=/path/to/tata` |
| `FileNotFoundError: Fast-dLLM v1 not found at …` | Fast-dLLM not cloned | clone NVlabs/Fast-dLLM into `external/`, or pass `--fast_dllm_path` |
| `FileNotFoundError: No cached samples found at …` | collect didn't run, or wrong cache dir | run collect; or fix `cfg.data.cache_root` |
| `FileNotFoundError: shard missing: …` | shards out of sync with `shards_manifest.json` | re-repack, or delete `shards/` + `shards_manifest.json` and start fresh |
| `KeyError: 'Could not find a prompt field in record …'` | Nemotron record schema mismatch | adjust `_default_prompt_extractor` in `collect_llada.py` |
| `RuntimeError: No LLaDA transformer blocks found` | wrapper class renamed in your Fast-dLLM checkout | update class names in `_find_last_block` |
| Startup shows `BlockShardSampler (legacy)` when you wanted `InterleavedShardSampler` | config pinned `shard_sampler: block`, or default changed unintentionally | check the config; for the current path use the `_interleaved_` config |
| Startup shows `i_ref-biased sampler:` for a shard cache | dataset didn't go into shard mode | check `shards_manifest.json` exists in `cache_root`; re-repack if not |
| wandb hangs at init | not logged in | `wandb login` once, then re-run |
| 401/403 from HuggingFace on Nemotron | gated dataset; not accepted | `huggingface-cli login` and accept on the dataset page |
| OOM during training | batch / preload / RAM | drop `data.batch_size`; drop `data.active_shards` (3-4) for the interleaved path |
| `RuntimeError: Error(s) in loading state_dict for VariantC` | resuming a pre-§3.2 checkpoint into the new architecture | start fresh — model definition changed (RoPE + RMSNorm + SwiGLU + per-pos conf) |
| `KeyError: 'block_start_pos'` or `'prefix_kv_pad_mask'` in train loop | stale `__pycache__` | `rm -rf delta_model/**/__pycache__` and re-run |
| `[time]` shows `data: >30%` after the first warm-up | shard LRU thrashing | bump `data.active_shards` (more shards resident), or `preload: true` if RAM allows |
| Bus error in DataLoader (SIGBUS) | cluster `/dev/shm` too small | configs already use `num_workers: 0`; if you flipped it back, revert |
| GSM8K mid-train eval fails | dtype mismatch in the rollback path | logged + skipped (training continues) |
| Sweep shows huge `mean_rollbacks` and tiny `mean_blocks_finished` at high threshold | pre-rollback-fix inference code | update to current `inference/hybrid_runner.py` (rollback commits vanilla Fast-dLLM tokens; engineering.md §5.1) |
| `ValueError: x and y must have same first dimension, but have shapes (0,) and (N,)` in `plot_metrics` | smoothing window > metric point count (older code) | update — current `plot_metrics.py` falls back to raw for sparse metrics |

---

## Cleanup / partial-data scenarios

Cache is fully resumable: collect skips files that already exist. Safe
to Ctrl-C and re-run.

To rebuild the test set from scratch (e.g. after changing
`subset_ratios` for the test partition), delete:

```bash
rm cache_v1_20k/llada/manifest.json
rm -rf cache_v1_20k/llada/test
```

The hash partition produces the same test IDs for any given subset
ratios, so you can rebuild the manifest without re-collecting if
`subset_ratios` for test stayed constant.

To revert a shard-mode cache back to per-sample (only possible if you
ran repack *without* `--rm_source`):

```bash
rm -rf cache_v1_20k/llada/shards
rm cache_v1_20k/llada/shards_manifest.json
```

If you want to keep a "broken cache" around as a reference while
working on a new one, move it aside rather than delete:

```bash
mv cache_v1_20k/llada  cache_v1_20k/llada_pre_thin
```

---

## What lives where

```
design.md         — research framing: problem, idea, prior work, milestones
engineering.md    — code-level contracts; recent-changes log; status of every component
usage.md          — this file: commands to run, error handling
delta_model/      — the importable package (see engineering.md §File map)
```

Old / historical configs (`m1_llada_variant_c.yaml`,
`m1_5_llada_variant_c.yaml`, `m1_6_*`, `m2_t5_*`, the spsv ablation)
are pinned with `val_split: holdout` and (for shard ones)
`shard_sampler: block` so re-running them reproduces their recorded
results. The current best is
`m1_5_data20k_interleaved_llada_variant_c.yaml`.
