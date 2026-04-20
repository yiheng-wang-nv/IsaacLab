#!/usr/bin/env bash
# ============================================================
# 并行 replay 多个 episode
# 用法: bash parallel_replay.sh "0,1,2,3,4,15,19,21,22" 0,1,2,3,5,6
# ============================================================
set -e

EP_LIST=${1:-"0"}
GPU_LIST=${2:-"0,1,2,3,5,6"}
DATASET_DIR=${3:-"/localhome/local-vennw/data/trocar_parallel_combined_success"}
OUTPUT_DIR=${4:-"/localhome/local-vennw/data/trocar_replay"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IFS=',' read -ra EPS <<< "$EP_LIST"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

mkdir -p "$OUTPUT_DIR"
echo "Replaying ${#EPS[@]} episodes on ${NUM_GPUS} GPUs"

# Distribute episodes round-robin to GPUs
declare -A GPU_EPS
for i in "${!EPS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_IDX]}
    GPU_EPS[$GPU_ID]="${GPU_EPS[$GPU_ID]} ${EPS[$i]}"
done

PIDS=()
for GPU_ID in "${!GPU_EPS[@]}"; do
    (
        for EP in ${GPU_EPS[$GPU_ID]}; do
            echo "  GPU $GPU_ID: episode $EP"
            bash "${SCRIPT_DIR}/replay_trocar.sh" $EP "$DATASET_DIR" "$OUTPUT_DIR" $GPU_ID \
                > "$OUTPUT_DIR/gpu_${GPU_ID}_ep_${EP}.log" 2>&1 || echo "  ep $EP FAILED"
        done
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait $pid; done
echo "All replays done. Videos in: $OUTPUT_DIR"
