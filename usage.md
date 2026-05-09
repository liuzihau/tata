# tata delta-model — running / smoke-testing

Operator's manual. Companion to `scoping.md` (design) and
`implementation_plan.md` (code structure). Use this doc when you're
about to run something on a GPU box.

`tata/` is **a standalone repo** — it does not import from any
sibling `peft_project/` tree. The only external code dependency is
**Fast-dLLM v1**, located at runtime via path / env var (see below).

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

## Build order — what to run, in order

All commands below assume cwd = **tata repo root**.

### 1) Compile-check (no torch needed)

```bash
python3 -m py_compile \
    delta_model/data/schema.py \
    delta_model/data/collect_llada.py \
    delta_model/data/dataset.py \
    delta_model/llada_runtime.py \
    delta_model/models/heads.py \
    delta_model/models/variant_c.py \
    delta_model/losses.py \
    delta_model/inference/hybrid_runner.py \
    delta_model/eval/shared_mass.py \
    delta_model/eval/gsm8k_e2e.py \
    delta_model/sanity/test_collect_roundtrip.py \
    delta_model/sanity/test_zero_init.py \
    delta_model/train.py \
  && echo OK
```

Pass criterion: prints `OK` (no syntax errors).

### 2) Zero-init sanity (CPU-only, no Fast-dLLM, no LLaDA)

Confirms the model + loss invariants before sinking any GPU time.

```bash
python -m delta_model.sanity.test_zero_init
```

Pass criterion: prints `✓ zero-init sanity passed`. The MSE term must
equal `((h_target - h_ref) ** 2).mean()` exactly (the "h_ref reuse"
baseline) and KL must be ≥ 0.

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

What to watch in wandb (project `tata-delta-model`, group `M1-llada`):
- `train/loss` decreasing from step 0 onward.
- `train/c_label_mean` rising from ~0.5 baseline toward 0.9+ as the
  delta model gets better.
- `val/mse_by_gap_{1..5}` — gap=1 should be much lower than gap=5.
- `gsm8k/accuracy_delta` — held-out subset every `gsm8k_every` steps;
  target by step 20000 is within 5pp of vanilla baseline.

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

4. **`prefix_proj` dim assumption.** `delta_model/models/variant_c.py`
   assumes `n_kv_heads * d_head == d_model`. True for LLaDA (no GQA).
   Dream (GQA factor 8) needs a small change to expand the KV bank
   or project from a different size; flagged for M3.

---

## File map (where things live)

```
tata/
  scoping.md                                 — design + locked decisions
  implementation_plan.md                     — file layout, shapes, M2 hooks
  usage.md                                   — this file
  delta_model/                               — the importable package
    llada_runtime.py                         — LLaDA + Fast-dLLM helpers (self-contained)
    data/
      schema.py                              — constants used everywhere
      collect_llada.py                       — cache builder
      dataset.py                             — TataDeltaDataset
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
      test_collect_roundtrip.py              — cache-format check
      test_zero_init.py                      — model + loss invariant check
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
