#!/usr/bin/env bash
# Run the five M1.5-v2 trainings sequentially.
#
# Each run writes to its own ckpt dir + wandb run; logs go to logs/v2/.
# Skip / select runs via SKIP="…" / ONLY="…" (space-separated short names).
#
# Examples:
#   ./run_v2_trainings.sh                                         # all 5
#   ONLY="5k_preload 10k_interleaved" ./run_v2_trainings.sh       # only those two
#   SKIP="10k_sample"                  ./run_v2_trainings.sh       # everything except the BlockShardSampler anchor
#
# Pre-flight: caches at the paths in each config exist + are thinned to
# 6 blocks × 4 iters × KV 32 on BOTH train/ and test/. See
# `usage.md` (test-thin workflow) for the exact thin commands.

set -euo pipefail

# --- config -----------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FAST_DLLM_PATH="${FAST_DLLM_PATH:-external/Fast-dLLM/v1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/v2}"

# short_name : config_filename
RUNS=(
  "5k_preload:m1_5_v2_5k_preload_llada_variant_c.yaml"
  "10k_preload:m1_5_v2_10k_preload_llada_variant_c.yaml"
  "10k_sample:m1_5_v2_10k_sample_llada_variant_c.yaml"
  "10k_interleaved:m1_5_v2_10k_interleaved_llada_variant_c.yaml"
  "20k_interleaved:m1_5_v2_20k_interleaved_llada_variant_c.yaml"
)

SKIP="${SKIP:-}"
ONLY="${ONLY:-}"

# --- helpers ----------------------------------------------------------------

in_list() {  # in_list <needle> <space-separated-haystack>
  local needle="$1"; local haystack="$2"
  for x in $haystack; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

should_run() {
  local short="$1"
  if [[ -n "$ONLY" ]] && ! in_list "$short" "$ONLY"; then return 1; fi
  if [[ -n "$SKIP" ]] &&    in_list "$short" "$SKIP"; then return 1; fi
  return 0
}

# --- preflight --------------------------------------------------------------

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

echo "[runner] repo=$REPO_ROOT fast_dllm=$FAST_DLLM_PATH log_dir=$LOG_DIR"
echo "[runner] selection: ONLY='$ONLY' SKIP='$SKIP'"

echo "[runner] py_compile preflight..."
python3 -m py_compile \
  delta_model/train.py \
  delta_model/inference/hybrid_runner.py \
  delta_model/data/schema.py \
  delta_model/data/collect_llada.py \
  delta_model/data/dataset.py \
  delta_model/data/repack.py \
  delta_model/data/thin_cache.py \
  delta_model/eval/gsm8k_e2e.py \
  delta_model/eval/plot_metrics.py \
  delta_model/eval/plot_sweep.py
echo "[runner] py_compile OK"

# --- main loop --------------------------------------------------------------

start_all=$SECONDS
n_ran=0; n_skipped=0; n_failed=0

for entry in "${RUNS[@]}"; do
  short="${entry%%:*}"
  config="${entry##*:}"
  cfg_path="delta_model/configs/${config}"

  if ! should_run "$short"; then
    echo "[runner] SKIP  ${short}"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [[ ! -f "$cfg_path" ]]; then
    echo "[runner] FAIL  ${short}: missing config ${cfg_path}" >&2
    n_failed=$((n_failed + 1))
    continue
  fi

  log_file="${LOG_DIR}/${short}.$(date +%Y%m%d_%H%M%S).log"
  echo "[runner] RUN   ${short}  cfg=${cfg_path}  log=${log_file}"
  start=$SECONDS
  if python -m delta_model.train \
        --config "$cfg_path" \
        --override "backbone.fast_dllm_path=${FAST_DLLM_PATH}" \
        2>&1 | tee "$log_file"; then
    dur=$((SECONDS - start))
    echo "[runner] DONE  ${short}  (${dur}s)"
    n_ran=$((n_ran + 1))
  else
    dur=$((SECONDS - start))
    echo "[runner] FAIL  ${short}  (${dur}s) — see ${log_file}" >&2
    n_failed=$((n_failed + 1))
    # Continue on failure so the rest still run; flip to `exit 1` if you want
    # the script to halt on the first failure.
  fi
done

total=$((SECONDS - start_all))
echo "[runner] all done. ran=${n_ran} skipped=${n_skipped} failed=${n_failed}  total=${total}s"
[[ $n_failed -gt 0 ]] && exit 1 || exit 0
