# tata — Think-Anchored Talk Adapter

> **tata** — *Think-Anchored Talk Adapter*: a lightweight 2-layer adapter
> that predicts per-iteration hidden-state corrections for masked-diffusion
> LMs (LLaDA-8B, Dream-7B), anchored to the backbone's first full forward
> so errors can't compound across iterations.

See `design.md` for *why*, `engineering.md` for *contracts*,
`usage.md` for the longer operator's manual.

```
data collect  →  thin  →  repack  →  train  →  GSM8K sweep  →  plots
                          (or rebuild manifest)              ↘  interactive gen
```

## Setup (one-time)

```bash
pip install -r requirements.txt
mkdir -p external && git clone https://github.com/NVlabs/Fast-dLLM external/Fast-dLLM
```

All commands assume cwd = repo root (`tata/`). Fast-dLLM v1 is discovered
via `--fast_dllm_path external/Fast-dLLM/v1` (CLI), `FAST_DLLM_V1_PATH`
(env), or the default relative path. Auth (`wandb login`,
`huggingface-cli login`) is covered in the Training section where each
is actually needed.

---

# Part 1 — Training

## 1. Collect cache

Runs LLaDA + Fast-dLLM in prefix-cache mode and writes one `.pt` per
prompt with `h_per_pass`, `reveal_per_pass`, and a 64-slot prefix-KV.
Nemotron Post-Training v2 is gated — `huggingface-cli login` once
before the first collect:

```bash
huggingface-cli login    # once per machine

python -m delta_model.data.collect_llada \
    --n_train 20000 --n_test 800 \
    --output_root cache_v1_20k/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Resumable (skips already-written files). Test split is hash-stable —
growing `--n_train` later never touches the same 800 test prompts.

## 2. Thin (optional, recommended)

In-place per-file thin to drop trailing blocks, cap iterations, shrink
KV window. Defaults match the v2 recipe (6 blocks × 4 iters × KV 32 on
train; test left full). Pass `--splits train,test` to align test too.

```bash
python -m delta_model.data.thin_cache \
    --cache_root cache_v1_20k/llada \
    --keep_blocks 6 --keep_iters 4 --prefix_window 32 \
    --splits train,test
```

## 3. Repack into shards

Required for `preload: false` configs (10k+ caches that don't fit in
RAM). `--rm_source` keeps peak disk at ~1× instead of ~2×.

```bash
python -m delta_model.data.repack \
    --cache_root cache_v1_20k/llada \
    --shard_size 50 --output_subdir shards --rm_source
# Re-pack one split without losing the other's manifest entries:
python -m delta_model.data.repack \
    --cache_root cache_v1_20k/llada --splits test \
    --shard_size 50 --output_subdir shards --rm_source --merge_manifest
```

If `shards_manifest.json` ever gets clobbered, rebuild it from on-disk
shards without re-packing:

```bash
python -m delta_model.data.rebuild_manifest --cache_root cache_v1_20k/llada
```

## 4. Train

`wandb login` once before the first run — training logs to the
`tata-delta-model` project and mirrors to a local `metrics.jsonl`.

The current frontier is the **v3 two-phase pipeline** (see `v3_plan.md`):
phase 1 trains the delta, phase 2 trains the confidence head on
free-running DAgger labels. The whole pipeline:

```bash
wandb login                          # once per machine
./run_v3.sh                          # phase 1 → top-3 → 3×(phase 2 + GSM8K sweep)
ONLY_PHASE1=1 ./run_v3.sh             # phase 1 only
SKIP_PHASE1=1 ./run_v3.sh             # phases 2+3 from existing phase-1 ckpts
```

Or run the phases by hand:

```bash
# Phase 1 — delta training (--phase delta forces lambda_conf=0,
# runs delta-only mid-train GSM8K, keeps the top-3 checkpoints).
python -m delta_model.train --phase delta \
    --config delta_model/configs/v3_phase1_delta_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1

# Phase 2 — conf-head training (frozen delta, free-running DAgger labels).
# Run once per top-3 phase-1 checkpoint, each to its own out_dir.
python -m delta_model.train --phase conf \
    --config delta_model/configs/v3_phase2_conf_llada_variant_c.yaml \
    --resume_from ckpts/v3_phase1_delta_llada_variant_c/best_delta_gsm8k_step<N>.pt \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1 \
    --override checkpoint.out_dir=ckpts/v3_phase2_conf_cand1
```

**Legacy / v2** — the v2 bracket configs (`m1_5_v2_*`, `m1_5_v2_t9/t10_*`)
still train with the default `--phase joint`:

```bash
python -m delta_model.train \
    --config delta_model/configs/m1_5_v2_20k_interleaved_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
./run_v2_trainings.sh                 # the full v2 bracket
```

Resume any run:

```bash
python -m delta_model.train --config <cfg> \
    --resume_from ckpts/<run>/step_0010000.pt
```

Checkpoints land in `ckpts/<run_name>/`:

| pattern | when | phase |
|---|---|---|
| `step_<NNNNNNN>.pt` | every `cfg.checkpoint.every` steps (rotated) | all |
| `best_val_kl_step<N>.pt` | new `val/kl` minimum | joint |
| `best_gsm8k_step<N>.pt` | new `gsm8k/accuracy_hybrid` max | joint |
| `best_delta_gsm8k_step<N>.pt` × 3 | top-3 by free-running delta-only GSM8K | delta (phase 1) |
| `best_conf_step<N>.pt` | best in-band (rollback rate ∈ [0.5,0.7]) | conf (phase 2) |

`<ckpt_dir>/metrics.jsonl` mirrors every `wandb.log` payload — plottable
locally without wandb cloud.

## 5. Plot training curves

```bash
# Single run (popup)
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_v2_20k_interleaved_llada_variant_c/metrics.jsonl

# Multi-run comparison on specific metrics
python -m delta_model.eval.plot_metrics \
    ckpts/m1_5_v2_10k_interleaved_llada_variant_c/metrics.jsonl \
    ckpts/m1_5_v2_20k_interleaved_llada_variant_c/metrics.jsonl \
    --metric train/kl val/kl val/bce gsm8k/accuracy_hybrid \
    --smooth 25 \
    --out plots/v2_data_scaling_interleaved.png

# Per-bin diagnostics (regex) — 1-D gap/reveal bins, the 2-D matrix,
# and the v3 diagnostics (val/bce, val/top1_agree)
python -m delta_model.eval.plot_metrics \
    ckpts/<run>/metrics.jsonl \
    --metric "val/mse_by_gap_[0-9]_reveal_[0-9]" \
    --out plots/gap_reveal_2d.png

python -m delta_model.eval.plot_metrics \
    ckpts/<run>/metrics.jsonl \
    --metric val/bce val/top1_agree gsm8k/accuracy_hybrid \
    --out plots/v3_diag.png
```

Metrics now available in `metrics.jsonl`: `train/kl`, `train/kl_backward`
(symmetric-KL runs), `val/kl`, `val/bce`, `val/top1_agree`,
`val/mse_by_gap_{g}`, `val/mse_by_reveal_{rb}`,
`val/mse_by_gap_{g}_reveal_{rb}` (+ `n_by_*` support counts),
`gsm8k/accuracy_hybrid`, `gsm8k/rollback_rate`, and (phase 1)
`gsm8k/*` from the delta-only eval.

---

# Part 2 — Inference

## Pre-trained checkpoints

If you don't want to train from scratch, grab a ready-made checkpoint
from:

**[Google Drive — tata checkpoints](https://drive.google.com/drive/folders/1dfjqqIJ_IviwzJz77rWGMyu0OhkvnnZE?usp=sharing)**

Currently published: **`best_val_kl_step7500.pt`** (one file). Drop it
into `ckpts/tata_release/`:

```bash
mkdir -p ckpts/tata_release
mv ~/Downloads/best_val_kl_step7500.pt ckpts/tata_release/
```

The inference commands in §A and §B below are temporarily pinned to
this exact filename. Once more checkpoints are released (e.g. a
`best_gsm8k_*` companion), they'll switch back to the
`best_<metric>_step*.pt` glob pattern so the latest of each kind is
picked automatically.

## A. GSM8K end-to-end evaluation (the headline metric)

Runs the hybrid (delta + Fast-dLLM) vs vanilla (Fast-dLLM only) on the
same GSM8K slice and sweeps `per_pos_threshold`. Reports per-threshold
accuracy, speedup, rollback rate, backbone calls, vanilla backbone
calls.

```bash
python -m delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/tata_release/best_val_kl_step7500.pt \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --factor 1.0 \
    --out_json eval_results/tata_release_sweep.json
```

Live `tqdm` shows running per-problem decode time hybrid vs vanilla,
running speedup, accuracy, hybrid/vanilla backbone calls, and rollback
rate — a broken threshold is visible after ~10 problems.

```
gsm8k thr=0.85:  47%|████▋ | 94/200  hyb=1.83s van=2.05s x=1.12
                                      acc_h=0.41 acc_v=0.75
                                      bb_h=40.3 bb_v=24.6 rb%=80
```

Three sweep modes (`gsm8k_e2e` flags):

```bash
# (1) fixed per_pos_threshold sweep — the default
python -m delta_model.eval.gsm8k_e2e --delta_ckpt <ckpt> \
    --fast_dllm_path external/Fast-dLLM/v1 --n_problems 200 \
    --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95 \
    --out_json eval_results/<run>_sweep.json

# (2) dynamic per-(gap,reveal) threshold lookup — needs a metrics.jsonl
#     with the 2-D val cells (any v3 / post-2026-05-19 run)
python -m delta_model.eval.gsm8k_e2e --delta_ckpt <ckpt> \
    --fast_dllm_path external/Fast-dLLM/v1 --n_problems 200 \
    --use_thr_lookup ckpts/<run>/metrics.jsonl \
    --thr_lookup_brackets "0.65:0.95,0.70:0.90" \
    --out_json eval_results/<run>_lookup_sweep.json

# (3) delta-only — pure free-running delta, no gate, no rollback
#     (the v3 phase-1 selection metric)
python -m delta_model.eval.gsm8k_e2e --delta_ckpt <ckpt> \
    --fast_dllm_path external/Fast-dLLM/v1 --n_problems 200 \
    --delta_only --out_json eval_results/<run>_deltaonly.json
```

Sweep every trained checkpoint sequentially with a runner script —
`run_v3.sh` runs the v3 sweeps as its stage 3; `run_v2_sweeps.sh`
covers the v2 bracket:

```bash
./run_v2_sweeps.sh
ONLY="t9_20k_interleaved" ./run_v2_sweeps.sh
N_PROBLEMS=50 PER_POS_THRESHOLDS="0.70,0.80,0.90" ./run_v2_sweeps.sh
CKPT_KIND=val_kl ./run_v2_sweeps.sh  # use best_val_kl checkpoints
```

Visualize a sweep (2×2 grid: Pareto, accuracy bars, speedup line,
rollback bars). Multiple files overlay:

```bash
# Single sweep
python -m delta_model.eval.plot_sweep \
    eval_results/v2_20k_interleaved_sweep.json \
    --out plots/v2_20k_interleaved_sweep.png

# Overlay 10k bracket
python -m delta_model.eval.plot_sweep \
    eval_results/10k_preload_sweep.json \
    eval_results/10k_sample_sweep.json \
    eval_results/10k_interleaved_sweep.json \
    --out plots/v2_10k_sweeps_overlay.png
```

## B. Generate text from a prompt (ad-hoc / interactive)

A ready-made standalone script lives at `inference.py` in the repo
root. It reads a question from stdin, decodes it twice — first through
tata's hybrid (`generate_with_delta`), then through vanilla Fast-dLLM
(`generate_with_prefix_cache`, same `factor` / mask) — and prints both
outputs side-by-side with wall-time, backbone-forward counts, rollbacks,
and the hybrid speedup ratio.

```bash
python inference.py
# Override the defaults via env vars:
DELTA_CKPT=ckpts/tata_release/best_val_kl_step7500.pt \
PER_POS_THRESHOLD=0.75 FACTOR=1.0 \
python inference.py
```

If you'd rather embed the call yourself, the underlying library
function is `generate_with_delta()` in
`delta_model/inference/hybrid_runner.py`. The minimum self-contained
wrapper:

```python
import torch
from delta_model.llada_runtime import load_llada
from delta_model.inference.hybrid_runner import generate_with_delta
from delta_model.models.variant_c import VariantC
import delta_model.data.schema as S

backbone, tokenizer = load_llada(fast_dllm_path="external/Fast-dLLM/v1")
ckpt = torch.load("ckpts/tata_release/best_val_kl_step7500.pt",
                  map_location="cpu", weights_only=False)
# Some LLaDA builds rename these config fields — fall back the same way
# train.py / gsm8k_e2e.py do (e.g. `layer_norm_eps` is absent on the
# checkpoint the user reported, only `layer_norm_type` is).
bb_cfg = {
    "rope_theta":  float(getattr(backbone.config, "rope_theta", 1e6)),
    "rms_eps":     float(getattr(backbone.config, "layer_norm_eps", 1e-5)),
    "max_seq_len": int(getattr(backbone.config, "max_sequence_length", 8192)),
}
delta = VariantC(d_model=ckpt["cfg"]["model"]["d_model"],
                 n_heads=ckpt["cfg"]["model"]["n_heads"],
                 n_layers=ckpt["cfg"]["model"]["n_layers"],
                 d_ff_inner=ckpt["cfg"]["model"].get("d_ff_inner"),
                 dropout=0.0,
                 detach_conf_features=bool(ckpt["cfg"]["model"]
                     .get("detach_conf_features", False)),
                 **bb_cfg).cuda().to(torch.bfloat16)
delta.load_state_dict(ckpt["model"], strict=True); delta.eval()

t = backbone.model.transformer
prompt = "Q: If a train leaves at 3pm and travels at 60mph, when does it arrive 180 miles away?\nA:"
msg = [{"role": "user", "content": prompt}]
ids = torch.tensor(tokenizer(tokenizer.apply_chat_template(
    msg, add_generation_prompt=True, tokenize=False))["input_ids"],
    dtype=torch.long).unsqueeze(0).cuda()

out_ids, stats = generate_with_delta(
    backbone, delta, t.ln_f, t.ff_out, t.wte,
    ids, per_pos_threshold=0.75, factor=1.0,
)
print(tokenizer.decode(out_ids[0, ids.shape[1]:].tolist(),
                       skip_special_tokens=True))
print(f"\n[stats] {stats['backbone_forwards']} backbone + "
      f"{stats['delta_forwards']} delta forwards, "
      f"{stats['rollbacks']} rollbacks, {stats['walltime']:.2f}s")
```

Knobs that affect quality vs speed:
- `per_pos_threshold` — lower → trust delta more (faster, possibly less
  accurate); higher → more rollbacks (slower, closer to vanilla). v2
  default 0.75–0.85.
- `factor` — Fast-dLLM dynamic-threshold scale. **Must match what
  collect used** (the cache's `manifest.json["decoding"]` records it).
- `inner_loop_max_iter` — None → `2·block_length` (=64 at BL=32); cap
  past-block budget.

## C. Sanity / one-off checks

| | command |
|---|---|
| Module compile | `python3 -m py_compile delta_model/{train,inference/hybrid_runner}.py delta_model/data/{schema,collect_llada,dataset,repack,thin_cache,rebuild_manifest}.py delta_model/eval/{gsm8k_e2e,plot_metrics,plot_sweep}.py` |
| Delta-head zero-init invariant | `python -m delta_model.sanity.test_zero_init` |
| Cache schema | `python -m delta_model.sanity.test_collect_roundtrip "cache_v1_20k/llada/train/sample_*.pt"` |
| InterleavedShardSampler | `python -m delta_model.sanity.test_interleaved_sampler` |
| Fast-dLLM partial vs full forward divergence (informational) | `python -m delta_model.sanity.test_partial_full_forward_equivalence --fast_dllm_path external/Fast-dLLM/v1` |

---

## Layout

```
tata/
├── README.md                                — this file
├── design.md  engineering.md  usage.md      — long-form docs
├── v3_plan.md                               — v3 implementation spec
├── run_v3.sh                                — v3 pipeline: phase1 → top-3 → 3×(phase2 + sweep)
├── run_v2_trainings.sh                      — v2-bracket sequential training runner
├── run_v2_sweeps.sh                         — v2-bracket sequential GSM8K sweep runner
├── inference.py                             — interactive hybrid-vs-vanilla single-prompt decode
├── delta_model/
│   ├── data/
│   │   ├── collect_llada.py                 — cache builder
│   │   ├── thin_cache.py                    — in-place thinner
│   │   ├── repack.py                        — per-sample → shard packer (+ --merge_manifest)
│   │   ├── rebuild_manifest.py              — regenerate shards_manifest.json from disk
│   │   ├── dataset.py                       — TataDeltaDataset + InterleavedShardSampler
│   │   └── schema.py                        — cache-format constants
│   ├── models/
│   │   ├── variant_c.py                     — VariantC (with detach_conf_features knob)
│   │   └── heads.py                         — DeltaHead, ConfHeadPerPos
│   ├── losses.py                            — composite MSE + KL(±backward) + tiered/shared-mass BCE
│   ├── train.py                             — train loop + --phase {joint,delta,conf}
│   ├── inference/
│   │   ├── hybrid_runner.py                 — generate_with_delta (+ delta_only mode)
│   │   ├── conf_rollout.py                  — v3 free-running DAgger labelling
│   │   └── thr_lookup.py                    — dynamic per-(gap,reveal) threshold
│   ├── eval/
│   │   ├── gsm8k_e2e.py                     — sweep harness (fixed / lookup / delta_only modes)
│   │   ├── plot_sweep.py                    — 2×2 sweep visualizer
│   │   └── plot_metrics.py                  — training curve overlay
│   ├── sanity/                              — T2..T5 tests
│   └── configs/
│       ├── v3_phase1_delta_llada_variant_c.yaml   — v3 phase 1 (delta)
│       ├── v3_phase2_conf_llada_variant_c.yaml    — v3 phase 2 (conf head)
│       ├── m1_5_v2_*_llada_variant_c.yaml         — v2 bracket + t9/t10 trials
│       └── configs_old/                            — legacy v1 configs
├── ckpts/                                   — checkpoints per run
├── eval_results/                            — sweep JSONs
├── plots/                                   — generated figures
└── external/Fast-dLLM/                      — vendored Fast-dLLM v1
```
