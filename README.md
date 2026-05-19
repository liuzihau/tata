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

`wandb login` once before the first run — every training script logs to
the `tata-delta-model` project there and also mirrors to a local
`metrics.jsonl`.

```bash
wandb login    # once per machine

python -m delta_model.train \
    --config delta_model/configs/m1_5_v2_20k_interleaved_llada_variant_c.yaml \
    --override backbone.fast_dllm_path=external/Fast-dLLM/v1
```

Resume:

```bash
python -m delta_model.train --config <cfg> \
    --resume_from ckpts/<run>/step_0010000.pt
```

Or run the whole v2 bracket (5k preload anchor + 10k×{preload, sample,
interleaved} + 20k×{preload, sample, interleaved} + T9) sequentially:

```bash
./run_v2_trainings.sh
# Subset:
ONLY="t9_20k_interleaved 20k_interleaved" ./run_v2_trainings.sh
SKIP="20k_preload" ./run_v2_trainings.sh
```

Per run, three kinds of checkpoints land in `ckpts/<run_name>/`:

| pattern | when | rotated by `keep_last` |
|---|---|---|
| `step_<NNNNNNN>.pt` | every `cfg.checkpoint.every` steps | yes |
| `best_val_kl_step<N>.pt` | new minimum on `val/kl` | no |
| `best_gsm8k_step<N>.pt` | new maximum on `gsm8k/accuracy_hybrid` | no |

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

# Per-bin diagnostics (regex)
python -m delta_model.eval.plot_metrics \
    ckpts/<run>/metrics.jsonl \
    --metric "val/mse_by_gap_.*" \
    --out plots/gap_bins.png
```

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

Sweep every trained v2 checkpoint sequentially with the runner script:

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

The hybrid decoder is a library function — `generate_with_delta()` in
`delta_model/inference/hybrid_runner.py`. No standalone CLI yet; the
canonical 25-line wrapper is:

```python
import torch
from delta_model.llada_runtime import load_llada
from delta_model.inference.hybrid_runner import generate_with_delta
from delta_model.models.variant_c import VariantC
import delta_model.data.schema as S

backbone, tokenizer = load_llada(fast_dllm_path="external/Fast-dLLM/v1")
ckpt = torch.load("ckpts/tata_release/best_val_kl_step7500.pt",
                  map_location="cpu", weights_only=False)
bb_cfg = {"rope_theta": float(backbone.config.rope_theta),
          "rms_eps":    float(backbone.config.layer_norm_eps),
          "max_seq_len": int(backbone.config.max_sequence_length)}
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
├── run_v2_trainings.sh                      — sequential training runner
├── run_v2_sweeps.sh                         — sequential GSM8K sweep runner
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
│   ├── losses.py                            — composite MSE + KL + per-pos BCE
│   ├── train.py                             — single-file training loop
│   ├── inference/
│   │   └── hybrid_runner.py                 — generate_with_delta
│   ├── eval/
│   │   ├── gsm8k_e2e.py                     — sweep harness (with rollback_rate, vanilla bb)
│   │   ├── plot_sweep.py                    — 2×2 sweep visualizer
│   │   └── plot_metrics.py                  — training curve overlay
│   ├── sanity/                              — T2..T5 tests
│   └── configs/
│       ├── m1_5_v2_*_llada_variant_c.yaml   — the active v2 bracket
│       ├── m1_5_v2_t9_*.yaml                — BCE-tame trial
│       └── configs_old/                     — legacy v1 configs
├── ckpts/                                   — checkpoints per run
├── eval_results/                            — sweep JSONs
├── plots/                                   — generated figures
└── external/Fast-dLLM/                      — vendored Fast-dLLM v1
```
