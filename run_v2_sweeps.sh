#!/usr/bin/env bash
# Sequentially run the gsm8k_e2e per_pos_threshold sweep against every
# trained v2 checkpoint. Mirrors run_v2_trainings.sh in shape: same
# short-name list, same ONLY / SKIP env-var selection, same per-run log
# directory layout.
#
# Per run:
#   • locate the best checkpoint in ckpts/<run_name>/  (best_gsm8k_* > best_val_kl_* > newest step_*.pt)
#   • run the sweep over PER_POS_THRESHOLDS on N_PROBLEMS gsm8k samples
#   • write JSON to eval_results/<short>_sweep.json
#   • write stdout/stderr to logs/v2_sweeps/<short>.<timestamp>.log
#
# Examples:
#   ./run_v2_sweeps.sh                                            # all 5
#   ONLY="5k_preload 10k_interleaved" ./run_v2_sweeps.sh          # only those two
#   SKIP="10k_sample"                  ./run_v2_sweeps.sh         # everything except the BlockShardSampler anchor
#   N_PROBLEMS=50  PER_POS_THRESHOLDS="0.80,0.85"  ./run_v2_sweeps.sh   # quick smoke
#   CKPT_KIND=val_kl                   ./run_v2_sweeps.sh         # use best_val_kl_* instead of best_gsm8k_*

set -euo pipefail

# --- config ----------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FAST_DLLM_PATH="${FAST_DLLM_PATH:-external/Fast-dLLM/v1}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/ckpts}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_results}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/v2_sweeps}"

N_PROBLEMS="${N_PROBLEMS:-200}"
PER_POS_THRESHOLDS="${PER_POS_THRESHOLDS:-0.70,0.80,0.85,0.90,0.95}"
FACTOR="${FACTOR:-1.0}"                       # must match what collect used
SEED="${SEED:-42}"
CKPT_KIND="${CKPT_KIND:-gsm8k}"               # gsm8k | val_kl

# short_name : ckpt-dir-basename (matches run_v2_trainings.sh + each config's checkpoint.out_dir)
RUNS=(
  "5k_preload:m1_5_v2_5k_preload_llada_variant_c"
  "10k_preload:m1_5_v2_10k_preload_llada_variant_c"
  "10k_sample:m1_5_v2_10k_sample_llada_variant_c"
  "10k_interleaved:m1_5_v2_10k_interleaved_llada_variant_c"
  "20k_interleaved:m1_5_v2_20k_interleaved_llada_variant_c"
  "20k_sample:m1_5_v2_20k_sample_llada_variant_c"
  # 20k_preload may not have a checkpoint if training OOM'd at startup —
  # find_ckpt logs FAIL and moves on, which is fine.
  "20k_preload:m1_5_v2_20k_preload_llada_variant_c"
)

SKIP="${SKIP:-}"
ONLY="${ONLY:-}"

# --- helpers ---------------------------------------------------------------

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

# Pick the best checkpoint for a run:
#   CKPT_KIND=gsm8k  → best_gsm8k_step*.pt   (preferred for downstream eval)
#   CKPT_KIND=val_kl → best_val_kl_step*.pt
# Falls back: requested → other "best" → newest rolling step_*.pt → fail.
find_ckpt() {
  local dir="$1"
  local primary secondary
  if [[ "$CKPT_KIND" == "val_kl" ]]; then
    primary="best_val_kl_step"; secondary="best_gsm8k_step"
  else
    primary="best_gsm8k_step";  secondary="best_val_kl_step"
  fi
  local hit
  hit=$(ls -1 "$dir/${primary}"*.pt 2>/dev/null | head -n 1 || true)
  if [[ -z "$hit" ]]; then
    hit=$(ls -1 "$dir/${secondary}"*.pt 2>/dev/null | head -n 1 || true)
  fi
  if [[ -z "$hit" ]]; then
    # Last resort: highest-numbered rolling step_*.pt.
    hit=$(ls -1 "$dir"/step_*.pt 2>/dev/null | sort | tail -n 1 || true)
  fi
  echo "$hit"
}

# --- preflight -------------------------------------------------------------

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[sweep] repo=$REPO_ROOT  fast_dllm=$FAST_DLLM_PATH"
echo "[sweep] ckpt_dir=$CKPT_DIR  out_dir=$OUT_DIR  log_dir=$LOG_DIR"
echo "[sweep] n_problems=$N_PROBLEMS  thresholds=$PER_POS_THRESHOLDS  factor=$FACTOR  ckpt_kind=$CKPT_KIND"
echo "[sweep] selection: ONLY='$ONLY' SKIP='$SKIP'"

echo "[sweep] py_compile preflight..."
python3 -m py_compile \
  delta_model/eval/gsm8k_e2e.py \
  delta_model/eval/plot_sweep.py \
  delta_model/inference/hybrid_runner.py
echo "[sweep] py_compile OK"

# --- main loop -------------------------------------------------------------

start_all=$SECONDS
n_ran=0; n_skipped=0; n_failed=0
ran_jsons=()

for entry in "${RUNS[@]}"; do
  short="${entry%%:*}"
  ckpt_name="${entry##*:}"
  ckpt_dir="$CKPT_DIR/$ckpt_name"

  if ! should_run "$short"; then
    echo "[sweep] SKIP  $short"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "[sweep] FAIL  $short: ckpt dir missing  ($ckpt_dir)" >&2
    n_failed=$((n_failed + 1))
    continue
  fi

  ckpt=$(find_ckpt "$ckpt_dir")
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "[sweep] FAIL  $short: no checkpoint found in $ckpt_dir" >&2
    n_failed=$((n_failed + 1))
    continue
  fi

  out_json="$OUT_DIR/${short}_sweep.json"
  log_file="$LOG_DIR/${short}.$(date +%Y%m%d_%H%M%S).log"

  echo "[sweep] RUN   $short  ckpt=$(basename "$ckpt")  out=$out_json  log=$log_file"
  start=$SECONDS
  if python -m delta_model.eval.gsm8k_e2e \
        --delta_ckpt "$ckpt" \
        --fast_dllm_path "$FAST_DLLM_PATH" \
        --n_problems "$N_PROBLEMS" \
        --per_pos_thresholds "$PER_POS_THRESHOLDS" \
        --factor "$FACTOR" \
        --seed "$SEED" \
        --out_json "$out_json" \
        2>&1 | tee "$log_file"; then
    dur=$((SECONDS - start))
    echo "[sweep] DONE  $short  (${dur}s)  → $out_json"
    ran_jsons+=("$out_json")
    n_ran=$((n_ran + 1))
  else
    dur=$((SECONDS - start))
    echo "[sweep] FAIL  $short  (${dur}s) — see $log_file" >&2
    n_failed=$((n_failed + 1))
    # Continue on failure so the queue still finishes. Flip to `exit 1` for
    # halt-on-first-failure semantics.
  fi
done

total=$((SECONDS - start_all))
echo "[sweep] all done. ran=$n_ran skipped=$n_skipped failed=$n_failed  total=${total}s"

# --- optional overlay plot of all sweeps that ran --------------------------

if [[ ${#ran_jsons[@]} -ge 1 ]]; then
  overlay_png="$OUT_DIR/v2_sweeps_overlay.png"
  echo "[sweep] overlay plot → $overlay_png"
  if ! python -m delta_model.eval.plot_sweep "${ran_jsons[@]}" --out "$overlay_png"; then
    echo "[sweep] overlay plot failed (non-fatal)." >&2
  fi
fi

[[ $n_failed -gt 0 ]] && exit 1 || exit 0
