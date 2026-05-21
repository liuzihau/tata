#!/usr/bin/env bash
# v3 pipeline runner: phase 1 → top-3 → 3×(phase 2 + GSM8K) → pick winner.
#
#   1. Phase 1 — delta training (--phase delta). Tracks the top-3
#      checkpoints by free-running delta-only GSM8K.
#   2. For each of the 3 phase-1 checkpoints: phase 2 — conf-head
#      training (--phase conf) with free-running DAgger labels.
#   3. GSM8K per_pos_threshold sweep on each phase-2 best_conf checkpoint.
#   4. Print the three sweeps' headline numbers — pick the winner.
#
# Stages are skippable for restarts:
#   SKIP_PHASE1=1 ./run_v3.sh      # phase 1 already done
#   ONLY_PHASE1=1 ./run_v3.sh      # stop after phase 1
#
# Env overrides: FAST_DLLM_PATH, N_PROBLEMS, PER_POS_THRESHOLDS.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FAST_DLLM_PATH="${FAST_DLLM_PATH:-external/Fast-dLLM/v1}"
N_PROBLEMS="${N_PROBLEMS:-200}"
PER_POS_THRESHOLDS="${PER_POS_THRESHOLDS:-0.70,0.80,0.85,0.90,0.95}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/v3}"

PHASE1_CFG="delta_model/configs/v3_phase1_delta_llada_variant_c.yaml"
PHASE2_CFG="delta_model/configs/v3_phase2_conf_llada_variant_c.yaml"
PHASE1_CKPT_DIR="${REPO_ROOT}/ckpts/v3_phase1_delta_llada_variant_c"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" eval_results

echo "[v3] repo=$REPO_ROOT  fast_dllm=$FAST_DLLM_PATH"

echo "[v3] py_compile preflight..."
python3 -m py_compile \
  delta_model/train.py \
  delta_model/losses.py \
  delta_model/inference/hybrid_runner.py \
  delta_model/inference/conf_rollout.py \
  delta_model/eval/gsm8k_e2e.py
echo "[v3] py_compile OK"

# --- Stage 1: phase-1 delta training ---------------------------------------

if [[ "${SKIP_PHASE1:-0}" != "1" ]]; then
  echo "[v3] === Stage 1: phase-1 delta training ==="
  python -m delta_model.train --phase delta \
      --config "$PHASE1_CFG" \
      --override "backbone.fast_dllm_path=${FAST_DLLM_PATH}" \
      2>&1 | tee "$LOG_DIR/phase1.$(date +%Y%m%d_%H%M%S).log"
else
  echo "[v3] SKIP_PHASE1=1 — using existing $PHASE1_CKPT_DIR"
fi

# top-3 phase-1 checkpoints (best_delta_gsm8k_step<N>.pt, written by the
# maybe_save_topk tracker).
mapfile -t PHASE1_CKPTS < <(ls -1 "$PHASE1_CKPT_DIR"/best_delta_gsm8k_step*.pt 2>/dev/null || true)
if [[ ${#PHASE1_CKPTS[@]} -eq 0 ]]; then
  echo "[v3] FAIL: no best_delta_gsm8k_step*.pt in $PHASE1_CKPT_DIR" >&2
  exit 1
fi
echo "[v3] phase-1 top-${#PHASE1_CKPTS[@]} checkpoints:"
printf '       %s\n' "${PHASE1_CKPTS[@]}"

if [[ "${ONLY_PHASE1:-0}" == "1" ]]; then
  echo "[v3] ONLY_PHASE1=1 — stopping after phase 1."
  exit 0
fi

# --- Stage 2+3: per-checkpoint phase-2 conf training + GSM8K sweep ---------

sweep_jsons=()
i=0
for p1 in "${PHASE1_CKPTS[@]}"; do
  i=$((i + 1))
  tag="cand${i}"
  p2_dir="${REPO_ROOT}/ckpts/v3_phase2_conf_${tag}"
  echo "[v3] === Stage 2 (${tag}): phase-2 conf training from $(basename "$p1") ==="
  python -m delta_model.train --phase conf \
      --config "$PHASE2_CFG" \
      --resume_from "$p1" \
      --override "backbone.fast_dllm_path=${FAST_DLLM_PATH}" \
      --override "checkpoint.out_dir=${p2_dir}" \
      2>&1 | tee "$LOG_DIR/phase2_${tag}.$(date +%Y%m%d_%H%M%S).log"

  # phase 2 keeps up to 2 admissible checkpoints (top-2 by speedup) —
  # sweep every one of them; Stage 3's full 200-problem sweep picks the
  # real winner between them.
  mapfile -t conf_ckpts < <(ls -1 "$p2_dir"/best_conf_step*.pt 2>/dev/null || true)
  if [[ ${#conf_ckpts[@]} -eq 0 ]]; then
    # no admissible checkpoint landed — fall back to the newest rolling one
    fallback=$(ls -1 "$p2_dir"/step_*.pt 2>/dev/null | sort | tail -n 1 || true)
    [[ -n "$fallback" ]] && conf_ckpts=("$fallback")
  fi
  if [[ ${#conf_ckpts[@]} -eq 0 ]]; then
    echo "[v3] WARN: no checkpoint produced for ${tag}; skipping its sweep" >&2
    continue
  fi

  ci=0
  for cc in "${conf_ckpts[@]}"; do
    ci=$((ci + 1))
    out_json="eval_results/v3_${tag}_c${ci}_sweep.json"
    echo "[v3] === Stage 3 (${tag} c${ci}): GSM8K sweep on $(basename "$cc") ==="
    python -m delta_model.eval.gsm8k_e2e \
        --delta_ckpt "$cc" \
        --fast_dllm_path "$FAST_DLLM_PATH" \
        --n_problems "$N_PROBLEMS" \
        --per_pos_thresholds "$PER_POS_THRESHOLDS" \
        --out_json "$out_json" \
        2>&1 | tee "$LOG_DIR/sweep_${tag}_c${ci}.$(date +%Y%m%d_%H%M%S).log"
    sweep_jsons+=("$out_json")
  done
done

# --- summary ---------------------------------------------------------------

echo "[v3] === done. per-candidate best hybrid accuracy ==="
for j in "${sweep_jsons[@]}"; do
  python3 -c "
import json, sys
rows = json.load(open('$j'))
best = max(rows, key=lambda r: r['accuracy_hybrid'])
print(f\"  $j : best acc_h={best['accuracy_hybrid']:.3f} \"
      f\"@ thr={best['per_pos_threshold']} speedup={best['speedup_ratio']:.2f}x\")
"
done

if [[ ${#sweep_jsons[@]} -ge 1 ]]; then
  echo "[v3] overlay plot → plots/v3_candidates_sweep.png"
  python -m delta_model.eval.plot_sweep "${sweep_jsons[@]}" \
      --out plots/v3_candidates_sweep.png || true
fi
