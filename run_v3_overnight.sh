#!/usr/bin/env bash
# Run the v3 pipeline, then hold the GPU with a keepalive so the cloud
# box does not auto-shutdown before you check results in the morning.
#
# The keepalive starts whether run_v3.sh succeeds OR fails — if training
# crashes overnight you still want the machine up to debug it (a restart
# costs ~2 h here).
#
# Launch under tmux / nohup so an SSH disconnect doesn't kill it:
#     nohup ./run_v3_overnight.sh > logs/overnight.out 2>&1 &
#   or
#     tmux new -s v3 './run_v3_overnight.sh'
#
# Env knobs:
#   KEEPALIVE_MIN   hard cap on the keepalive, minutes (default 720 = 12 h)
#   RUN_CMD         the job to run before the keepalive (default ./run_v3.sh)
#
# In the morning: check the logs, then Ctrl-C (or `kill`) the keepalive.

# NOT `set -e` — a run_v3 failure must still fall through to the keepalive.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/overnight.$ts.log"

RUN_CMD="${RUN_CMD:-./run_v3.sh}"
KEEPALIVE_MIN="${KEEPALIVE_MIN:-720}"

echo "[overnight] $(date) — starting: $RUN_CMD" | tee -a "$log"
bash -c "$RUN_CMD" 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
echo "[overnight] $(date) — job exited rc=$rc" | tee -a "$log"

echo "[overnight] $(date) — GPU keepalive up to ${KEEPALIVE_MIN} min "\
"(Ctrl-C / kill to stop)" | tee -a "$log"
python gpu_keepalive.py --minutes "$KEEPALIVE_MIN" 2>&1 | tee -a "$log"

echo "[overnight] $(date) — keepalive ended. machine free to idle." | tee -a "$log"
exit "$rc"
