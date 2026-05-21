#!/usr/bin/env bash
set -euo pipefail

OPEN_LOOP_STEPS_LIST=(1 2 4 8)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

for open_loop_steps in "${OPEN_LOOP_STEPS_LIST[@]}"; do
    echo "[BENCHMARK] Running threshold/timeout hybrid with open_loop_steps=${open_loop_steps}"
    bash ./check_multistage_direct.sh "${open_loop_steps}"

    echo "[BENCHMARK] Running timeout-only baseline with open_loop_steps=${open_loop_steps}"
    bash ./check_multistage_direct_timeout_only.sh "${open_loop_steps}"
done

echo "[BENCHMARK] Completed all open_loop_steps: ${OPEN_LOOP_STEPS_LIST[*]}"

