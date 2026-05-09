# tata delta-model — running / smoke-testing

Operator's manual. Companion to `scoping.md` (design) and
`implementation_plan.md` (code structure). Use this doc when you're
about to run something on a GPU box.

---

## Prereqs

- `huggingface-cli login` once on the collect machine
  (Nemotron Post-Training v2 is gated).
- `pip install` whatever's already in `probe_runner/requirements.txt`,
  plus `wandb` and `pyyaml`.
- `wandb login` once on the train machine.
- Fast-dLLM v1 reachable at `external/Fast-dLLM/v1`
  (or set `$FAST_DLLM_V1_PATH`, or pass `--fast_dllm_path`).
- Working dir for all commands is the repo root that contains
  `peft_project/`.

---

## Build order — what to run, in order

### 1) Tiny smoke test (≈ 2-3 min)

Collects 2 samples, then runs the cache-format roundtrip check.

```bash
python -m peft_project.tata.delta_model.data.collect_llada \
    --limit 2 --output_root cache_v1/llada_smoke \
    --fast_dllm_path external/Fast-dLLM/v1

python -m peft_project.tata.delta_model.sanity.test_collect_roundtrip \
    cache_v1/llada_smoke/test/sample_*.pt
```

Pass criteria:
- `[OK ]` for every sample (no shape / dtype mismatches, monotone reveal).
- The collect run prints `[collect] N XXX … done in YYs (mean YYs)`
  with no `FAILED` lines.

### 2) Zero-init baseline (CPU-friendly, ≈ 5 sec)

Confirms the model + loss invariants before sinking any GPU time.

```bash
python -m peft_project.tata.delta_model.sanity.test_zero_init
```

Pass criteria: prints `✓ zero-init sanity passed`. The MSE term must
equal `((h_target - h_ref) ** 2).mean()` exactly (the "h_ref reuse"
baseline) and KL must be ≥ 0.

### 3) Full data collection (≈ 4-7 hours on A100)

Produces `cache_v1/llada/{train,test}/sample_XXXXXXXX.pt` plus
`cache_v1/llada/manifest.json`. The test split is **hash-stable** —
re-running with a larger `--n_train` will not touch the same 800 test
samples, so you can grow the train set across runs without
contaminating eval.

```bash
huggingface-cli login    # one-time

python -m peft_project.tata.delta_model.data.collect_llada \
    --n_train 5000 --n_test 800 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1
```

Optional knobs:
- `--subset_ratios '{"chat":0.4,"math":0.4,"code":0.2,"stem":0.0}'`
  — math-heavier mix.
- `--threshold 0.85` — looser parallel-decoding threshold (more
  passes per block, more `(i_ref, i_target)` pairs per sample).
- `--start_index N` to resume a partially-done run.

Storage: ~80 GB at default 5800 samples (5000 train + 800 test).

### 4) Train (≈ 3 hours on A100 for 20k steps)

```bash
python -m peft_project.tata.delta_model.train \
    --config peft_project/tata/delta_model/configs/m1_llada_variant_c.yaml
```

Single-line config overrides via `--override`:

```bash
python -m peft_project.tata.delta_model.train \
    --config peft_project/tata/delta_model/configs/m1_llada_variant_c.yaml \
    --override data.batch_size=128 \
    --override optim.lr=5e-5 \
    --override optim.max_steps=10000
```

Resume from a checkpoint:

```bash
python -m peft_project.tata.delta_model.train \
    --config peft_project/tata/delta_model/configs/m1_llada_variant_c.yaml \
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

### 5) Final eval — sweep conf-threshold

```bash
python -m peft_project.tata.delta_model.eval.gsm8k_e2e \
    --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \
    --n_problems 200 \
    --conf_thresholds 0.80,0.85,0.90,0.95 \
    --out_json eval_results/m1_v0.json
```

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

1. **Nemotron v2 record schema.** `data/collect_llada.py:_default_prompt_extractor`
   tries keys `messages` → `input` → `prompt` → `question` → `query`
   → `instruction` and falls back to a clear `KeyError`. If none
   match, edit the function or pass a custom extractor.

2. **LLaDA wrapper paths.** In `train.py:_load_backbone_components`
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
   above (step 1) catches gross issues. For deeper verification:
   diff the generated tokens from collect vs. running Fast-dLLM
   directly on the same prompt — should match exactly at the same
   `--threshold`.

4. **`prefix_proj` dim assumption.** `models/variant_c.py` assumes
   `n_kv_heads * d_head == d_model`. True for LLaDA (no GQA). Dream
   (GQA factor 8) needs a small change to expand the KV bank or
   project from a different size; flagged for M3.

5. **wandb auth.** First time on a fresh box, `wandb login` will
   prompt for an API key. The training process won't progress past
   `_setup_wandb` until you've done this.

---

## File map (where things live)

```
peft_project/tata/
  scoping.md                                 — design + locked decisions
  implementation_plan.md                     — file layout, shapes, M2 hooks
  usage.md                                   — this file
  delta_model/
    data/schema.py                           — constants used everywhere
    data/collect_llada.py                    — cache builder
    data/dataset.py                          — TataDeltaDataset
    models/heads.py                          — Δh head (zero-init), conf head
    models/variant_c.py                      — M1 model
    losses.py                                — composite MSE + KL + BCE
    train.py                                 — AdamW + cosine + wandb
    inference/hybrid_runner.py               — generate_with_delta + rollback
    eval/shared_mass.py                      — overlap metric
    eval/gsm8k_e2e.py                        — end-to-end eval harness
    sanity/test_collect_roundtrip.py         — cache-format check
    sanity/test_zero_init.py                 — model + loss invariant check
    configs/m1_llada_variant_c.yaml          — default M1 config
```

## Common error -> fix table

| symptom | likely cause | fix |
|---|---|---|
| `FileNotFoundError: No cached samples found at cache_v1/llada/train` | collect didn't run, or path mismatch | re-run collect, or update `cfg.data.cache_root` |
| `KeyError: 'Could not find a prompt field in record …'` | Nemotron schema mismatch | adjust `_default_prompt_extractor` in `collect_llada.py` |
| `RuntimeError: No LLaDA transformer blocks found` | model wrapping changed | update class names in `_find_last_block` (collect + hybrid_runner) |
| `RuntimeError: prompt too short (< 32 tokens)` skips ALL samples | prompts are unusually short | lower `S.PREFIX_WINDOW`, or filter such prompts upstream |
| wandb hangs at init | not logged in | `wandb login` once, then re-run |
| OOM during training | batch too big for the GPU | `--override data.batch_size=128` (or 64) |
| OOM during collect | LLaDA + cache snapshots peak | reduce concurrent samples / restart, or rebuild with `--threshold 0.85` (faster blocks → less peak) |
| GSM8K mid-train eval fails | usually a rollback path bug under bf16 dtype mismatch | logged + skipped (training continues), check the printed traceback in the run log |

## Cleanup / partial-data scenarios

Cache is fully resumable: collect skips files that already exist
(`if out_path.exists(): continue`). Safe to Ctrl-C and re-run.

To rebuild test set from scratch (e.g. after changing
`subset_ratios` for the test partition), delete:

```bash
rm cache_v1/llada/manifest.json
rm -rf cache_v1/llada/test
```

The hash partition will produce the same test IDs for any given
subset ratios, so you can also just rebuild the manifest without
re-collecting if the per-subset ratios for test stayed constant.
