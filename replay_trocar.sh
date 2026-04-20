#!/usr/bin/env bash
# ============================================================
# 录制 episode 的 replay 验证
# 用法:
#   bash replay_trocar.sh <episode_idx> [dataset_dir] [output_dir] [gpu_id]
# ============================================================
set -e

EP_IDX=${1:-0}
DATASET_DIR=${2:-"/localhome/local-vennw/data/trocar_parallel/merged"}
OUTPUT_DIR=${3:-"/localhome/local-vennw/data/trocar_replay"}
GPU_ID=${4:-0}

CONDA_DIR="/localhome/local-vennw/miniconda3"
ISAACLAB_DIR="/localhome/local-vennw/code/IsaacLab"
ISAACSIM_DIR="/localhome/local-vennw/isaac-sim-standalone-6.0.0-rc.22"

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

cd "$ISAACLAB_DIR"

./isaaclab.sh -p scripts/tools/replay_trocar_episodes.py \
    --dataset_dir "$DATASET_DIR" \
    --episode_idx $EP_IDX \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:$GPU_ID \
    --randomize_lighting
