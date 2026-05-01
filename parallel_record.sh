#!/usr/bin/env bash
# ============================================================
# 多 GPU 并行录制 Trocar episodes
# 用法:
#   bash parallel_record.sh [总episode数] [GPU列表(逗号分隔)] [skip_first_n] [rand_light] [post_process] [output_dir] [open_loop_steps] [额外record参数...]
# 例:
#   bash parallel_record.sh 100 0,1,2,3 3 1 both /tmp/out 1 --model_path /path/to/ckpt
#   BASE_SEED=123456 bash parallel_record.sh 800 4,5,6,7
# post_process: none / success / split / both (default: both = success + split-by-stage)
# ============================================================
set -e
set -o pipefail

START_TIME=$(date +%s)

TOTAL_EPISODES=${1:-100}
GPU_LIST=${2:-"0,1,2,3"}
SKIP_FIRST_N=${3:-3}
RANDOMIZE_LIGHTING=${4:-1}
POST_PROCESS=${5:-both}  # none | success | split | both
BASE_OUTPUT_DIR=${6:-"/localhome/local-vennw/code/trocar_parallel_1_steps_default_rand_light_fix_seed_issue"}
OPEN_LOOP_STEPS=${7:-1}
EXTRA_ARGS=("${@:8}")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_DIR=${CONDA_DIR:-"/localhome/local-vennw/miniconda3"}
ISAACLAB_DIR=${ISAACLAB_DIR:-"/localhome/local-vennw/code/IsaacLab"}
ISAACSIM_DIR=${ISAACSIM_DIR:-"/localhome/local-vennw/isaac-sim-standalone-6.0.0-rc.22"}
MODEL_PATH=${MODEL_PATH:-"/localhome/local-vennw/models/orca_rlinf_weights/rlinf/actor/model_state_dict"}
MAX_STEPS=${MAX_STEPS:-300}
DENOISING_STEPS=${DENOISING_STEPS:-4}
SKIP_LAST_N=${SKIP_LAST_N:-1}
TRAY_YAW_MIN_DEG=${TRAY_YAW_MIN_DEG:-0}
TRAY_YAW_MAX_DEG=${TRAY_YAW_MAX_DEG:-10}
BASE_SEED=${BASE_SEED:-$(date +%s)}
PY="/localhome/local-vennw/miniconda3/envs/isaaclab_develop_6.0/bin/python"

# ---- IsaacLab / Isaac Sim environment ----
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate isaaclab_develop_6.0

export ISAAC_PATH="$ISAACSIM_DIR"
export EXP_PATH="$ISAAC_PATH/apps"
export CARB_APP_PATH="$ISAAC_PATH/kit"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
unset DISPLAY
export MESA_GL_VERSION_OVERRIDE=4.6

# Parse GPU list into array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

# 每个 GPU 分配的 episode 数
PER_GPU=$((TOTAL_EPISODES / NUM_GPUS))
REMAINDER=$((TOTAL_EPISODES % NUM_GPUS))

echo "============================================================"
echo "Parallel recording: ${TOTAL_EPISODES} episodes on ${NUM_GPUS} GPUs"
echo "  GPUs: ${GPU_LIST}"
echo "  Per GPU: ${PER_GPU} episodes (+ ${REMAINDER} extra on first GPUs)"
echo "  Skip first N frames: ${SKIP_FIRST_N}"
echo "  Skip last N frames: ${SKIP_LAST_N}"
echo "  Open-loop steps: ${OPEN_LOOP_STEPS}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Randomize lighting: ${RANDOMIZE_LIGHTING}"
echo "  Tray yaw [deg]: ${TRAY_YAW_MIN_DEG} .. ${TRAY_YAW_MAX_DEG}"
echo "  Base seed: ${BASE_SEED}"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${BASE_OUTPUT_DIR}"
echo "============================================================"

mkdir -p "$BASE_OUTPUT_DIR"
cd "$ISAACLAB_DIR"

PIDS=()
PID_GPUS=()
PID_EXPECTED=()
for IDX in "${!GPU_ARRAY[@]}"; do
    GPU_ID=${GPU_ARRAY[$IDX]}

    # Distribute remainder to first few GPUs
    EP_COUNT=$PER_GPU
    if [ $IDX -lt $REMAINDER ]; then
        EP_COUNT=$((PER_GPU + 1))
    fi

    if [ $EP_COUNT -eq 0 ]; then
        continue
    fi

    GPU_OUTPUT="${BASE_OUTPUT_DIR}/gpu_${GPU_ID}"
    LOG_FILE="${BASE_OUTPUT_DIR}/gpu_${GPU_ID}.log"

    SEED=$((BASE_SEED + GPU_ID * 1000 + IDX))
    echo "  GPU ${GPU_ID}: ${EP_COUNT} episodes → ${GPU_OUTPUT} (seed ${SEED}, visible as cuda:0 inside worker)"
    # IsaacLab Fabric/Warp paths used by this task only support cuda:0 reliably.
    # Isolate each worker to one physical GPU, then use cuda:0 inside the process.
    LIGHT_FLAG=""
    if [ "$RANDOMIZE_LIGHTING" = "1" ]; then
        LIGHT_FLAG="--randomize_lighting"
    fi
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" ./isaaclab.sh -p scripts/tools/record_trocar_episodes.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$GPU_OUTPUT" \
        --num_envs 1 \
        --num_episodes $EP_COUNT \
        --max_steps "$MAX_STEPS" \
        --denoising_steps "$DENOISING_STEPS" \
        --seed $SEED \
        --skip_first_n $SKIP_FIRST_N \
        --skip_last_n "$SKIP_LAST_N" \
        --device cuda:0 \
        --model_device cuda:0 \
        --open_loop_steps $OPEN_LOOP_STEPS \
        --tray_yaw_min_deg "$TRAY_YAW_MIN_DEG" \
        --tray_yaw_max_deg "$TRAY_YAW_MAX_DEG" \
        $LIGHT_FLAG \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    PID_GPUS+=("$GPU_ID")
    PID_EXPECTED+=("$EP_COUNT")
done

echo ""
echo "All ${#PIDS[@]} processes launched. Waiting..."
echo "Monitor: watch -n 5 'for d in ${BASE_OUTPUT_DIR}/gpu_*/data/chunk-000; do echo \$d: \$(ls \$d 2>/dev/null | wc -l) episodes; done'"
echo ""

# Wait and report
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "  GPU ${PID_GPUS[$i]}: process exited"
    else
        echo "  GPU ${PID_GPUS[$i]}: FAILED (see ${BASE_OUTPUT_DIR}/gpu_${PID_GPUS[$i]}.log)"
        FAILED=$((FAILED + 1))
    fi
done

for i in "${!PID_GPUS[@]}"; do
    GPU_ID=${PID_GPUS[$i]}
    EXPECTED=${PID_EXPECTED[$i]}
    GPU_OUTPUT="${BASE_OUTPUT_DIR}/gpu_${GPU_ID}"
    LOG_FILE="${BASE_OUTPUT_DIR}/gpu_${GPU_ID}.log"
    EP_DIR="${GPU_OUTPUT}/data/chunk-000"
    if [ -d "$EP_DIR" ]; then
        EP_WRITTEN=$(find "$EP_DIR" -maxdepth 1 -type f -name 'episode_*.parquet' | wc -l)
    else
        EP_WRITTEN=0
    fi
    if [ "$EP_WRITTEN" -ne "$EXPECTED" ] || [ ! -f "${GPU_OUTPUT}/episode_results.json" ]; then
        echo "  GPU ${GPU_ID}: INVALID output (${EP_WRITTEN}/${EXPECTED} episodes; see ${LOG_FILE})"
        FAILED=$((FAILED + 1))
    else
        echo "  GPU ${GPU_ID}: wrote ${EP_WRITTEN}/${EXPECTED} episodes"
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: ${FAILED} worker/output validation failure(s). Check logs."
    exit 1
fi

echo ""
echo "============================================================"
echo "All done! Merging..."
echo "============================================================"

# Merge: renumber episodes and combine into single output
MERGED_DIR="${BASE_OUTPUT_DIR}/merged"
$PY -c "
import json, shutil, os
from pathlib import Path

base = Path('${BASE_OUTPUT_DIR}')
merged = Path('${MERGED_DIR}')
merged.mkdir(parents=True, exist_ok=True)

ep_idx = 0
all_results = []
all_lengths = []
gpu_dirs = sorted(p for p in base.glob('gpu_*') if p.is_dir())

for gpu_dir in gpu_dirs:
    data_dir = gpu_dir / 'data/chunk-000'
    if not data_dir.exists():
        continue

    results_file = gpu_dir / 'episode_results.json'
    gpu_results = []
    if results_file.exists():
        with open(results_file) as f:
            gpu_results = json.load(f).get('episodes', [])

    for ep_file in sorted(data_dir.glob('episode_*.parquet')):
        old_idx = int(ep_file.stem.split('_')[1])

        out_data = merged / 'data/chunk-000'
        out_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ep_file, out_data / f'episode_{ep_idx:06d}.parquet')

        for subdir in ['videos', 'masks']:
            for cam_dir in sorted((gpu_dir / subdir / 'chunk-000').glob('*')):
                out_cam = merged / subdir / 'chunk-000' / cam_dir.name
                out_cam.mkdir(parents=True, exist_ok=True)
                for f in cam_dir.glob(f'episode_{old_idx:06d}*'):
                    new_name = f.name.replace(f'episode_{old_idx:06d}', f'episode_{ep_idx:06d}')
                    shutil.copy2(f, out_cam / new_name)

        for r in gpu_results:
            if r['episode_index'] == old_idx:
                r_copy = r.copy()
                r_copy['episode_index'] = ep_idx
                all_results.append(r_copy)
                all_lengths.append(r['length'])
                break

        ep_idx += 1

with open(merged / 'episode_results.json', 'w') as f:
    json.dump({
        'total_episodes': len(all_results),
        'success_count': sum(1 for r in all_results if r.get('success')),
        'fail_count': sum(1 for r in all_results if not r.get('success')),
        'episodes': all_results,
    }, f, indent=2)

if not gpu_dirs:
    raise RuntimeError(f'No gpu_* output directories found under {base}')

first_gpu = gpu_dirs[0]
for f in ['category_mapping.json']:
    src = first_gpu / f
    if src.exists():
        shutil.copy2(src, merged / f)

# Regenerate meta files from merged dataset
import numpy as np
import pandas as pd

meta_dir = merged / 'meta'
meta_dir.mkdir(parents=True, exist_ok=True)

# Copy info.json/modality.json/tasks.jsonl from first GPU (static content)
for fname in ['info.json', 'modality.json', 'tasks.jsonl']:
    src = first_gpu / 'meta' / fname
    if src.exists():
        shutil.copy2(src, meta_dir / fname)

# Regenerate episodes.jsonl for merged set
merged_data_dir = merged / 'data/chunk-000'
total_frames = 0
ep_lengths = []
with open(meta_dir / 'episodes.jsonl', 'w') as f:
    for ep_file in sorted(merged_data_dir.glob('episode_*.parquet')):
        ep_i = int(ep_file.stem.split('_')[1])
        df = pd.read_parquet(ep_file)
        length = len(df)
        ep_lengths.append(length)
        total_frames += length
        # find task from first result
        task_desc = 'install trocar from box'
        for r in all_results:
            if r['episode_index'] == ep_i:
                break
        f.write(json.dumps({'episode_index': ep_i, 'tasks': [task_desc], 'length': length}) + '\n')

# Update info.json with correct totals
info_path = meta_dir / 'info.json'
if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    info['total_episodes'] = len(ep_lengths)
    info['total_frames'] = total_frames
    info['total_videos'] = len(ep_lengths) * 3
    info['splits'] = {'train': f'0:{len(ep_lengths)}'}
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

# Regenerate episodes_stats.jsonl + stats.json
def _field_stats(arr):
    return {
        'min': arr.min(axis=0).tolist(),
        'max': arr.max(axis=0).tolist(),
        'mean': arr.mean(axis=0).tolist(),
        'std': arr.std(axis=0).tolist(),
        'q01': np.quantile(arr, 0.01, axis=0).tolist(),
        'q99': np.quantile(arr, 0.99, axis=0).tolist(),
    }

all_states = []
all_actions = []
per_ep_stats = []
for ep_file in sorted(merged_data_dir.glob('episode_*.parquet')):
    ep_i = int(ep_file.stem.split('_')[1])
    df = pd.read_parquet(ep_file)
    states_arr = np.array([np.asarray(s) for s in df['observation.state']], dtype=np.float32)
    actions_arr = np.array([np.asarray(a) for a in df['action']], dtype=np.float32)
    all_states.append(states_arr)
    all_actions.append(actions_arr)
    per_ep_stats.append({
        'episode_index': ep_i,
        'stats': {
            'observation.state': _field_stats(states_arr),
            'action': _field_stats(actions_arr),
        },
    })

with open(meta_dir / 'episodes_stats.jsonl', 'w') as f:
    for s in per_ep_stats:
        f.write(json.dumps(s) + '\n')

if all_states:
    with open(meta_dir / 'stats.json', 'w') as f:
        json.dump({
            'observation.state': _field_stats(np.concatenate(all_states, axis=0)),
            'action': _field_stats(np.concatenate(all_actions, axis=0)),
        }, f, indent=2)

print(f'Merged {ep_idx} episodes into {merged}')
print(f'Success: {sum(1 for r in all_results if r.get(\"success\"))}/{len(all_results)}')
"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo "Merged data: ${MERGED_DIR}"

# Post-processing
ISAACLAB_DIR="$(dirname "$SCRIPT_DIR")/IsaacLab"
# When parallel_record.sh is inside the IsaacLab directory, use $SCRIPT_DIR directly
ISAACLAB_DIR="$SCRIPT_DIR"

if [ "$POST_PROCESS" = "success" ] || [ "$POST_PROCESS" = "both" ]; then
    SUCCESS_DIR="${BASE_OUTPUT_DIR}/success_only"
    echo "Filtering success episodes → ${SUCCESS_DIR}"
    $PY "${ISAACLAB_DIR}/scripts/tools/filter_success_episodes.py" \
        --input_dir "$MERGED_DIR" \
        --output_dir "$SUCCESS_DIR"
fi

if [ "$POST_PROCESS" = "split" ] || [ "$POST_PROCESS" = "both" ]; then
    SPLIT_INPUT="$MERGED_DIR"
    if [ "$POST_PROCESS" = "both" ] && [ -d "${BASE_OUTPUT_DIR}/success_only" ]; then
        SPLIT_INPUT="${BASE_OUTPUT_DIR}/success_only"
    fi
    SPLIT_DIR="${BASE_OUTPUT_DIR}/split_by_stage"
    echo "Splitting by stage → ${SPLIT_DIR}"
    $PY "${ISAACLAB_DIR}/scripts/tools/split_by_stage.py" \
        --input_dir "$SPLIT_INPUT" \
        --output_dir "$SPLIT_DIR"
fi

echo "Total time: ${MINS}m ${SECS}s (${ELAPSED}s)"
echo "============================================================"
