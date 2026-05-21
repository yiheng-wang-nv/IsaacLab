#!/usr/bin/env bash
set -euo pipefail

OPEN_LOOP_STEPS_LIST=(1 2 4 8)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

is_complete() {
    local video_dir="$1"
    python - "$video_dir" <<'PY'
import json
import sys
from pathlib import Path

success_path = Path(sys.argv[1]) / "success.json"
if not success_path.exists():
    sys.exit(1)

try:
    data = json.loads(success_path.read_text())
except Exception:
    sys.exit(1)

episodes = data.get("episodes", [])
if data.get("completed_episodes") == 100 and len(episodes) == 100:
    sys.exit(0)
sys.exit(1)
PY
}

prepare_incomplete_dir() {
    local video_dir="$1"
    if [[ -d "${video_dir}" ]] && ! is_complete "${video_dir}"; then
        local backup="${video_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
        echo "[BENCHMARK] Existing incomplete dir found; moving to ${backup}"
        mv "${video_dir}" "${backup}"
    fi
}

run_if_needed() {
    local label="$1"
    local open_loop_steps="$2"
    local video_dir="$3"
    shift 3

    if is_complete "${video_dir}"; then
        echo "[BENCHMARK] Skipping completed ${label} with open_loop_steps=${open_loop_steps}: ${video_dir}"
        return
    fi

    prepare_incomplete_dir "${video_dir}"
    echo "[BENCHMARK] Running ${label} with open_loop_steps=${open_loop_steps}"
    "$@" "${open_loop_steps}"
}

for open_loop_steps in "${OPEN_LOOP_STEPS_LIST[@]}"; do
    hybrid_dir="${SCRIPT_DIR}/multistage_direct_videos_100e_t1_t2_timeout_t3_99997_t4_99_t5_env_success_rand_0_10_chunk${open_loop_steps}"
    timeout_dir="${SCRIPT_DIR}/multistage_direct_videos_100e_timeout_only_60_rand_0_10_chunk${open_loop_steps}"

    run_if_needed "threshold/timeout hybrid" "${open_loop_steps}" "${hybrid_dir}" bash ./check_multistage_direct.sh
    run_if_needed "timeout-only baseline" "${open_loop_steps}" "${timeout_dir}" bash ./check_multistage_direct_timeout_only.sh
done

echo "[BENCHMARK] Resume run completed for open_loop_steps: ${OPEN_LOOP_STEPS_LIST[*]}"
