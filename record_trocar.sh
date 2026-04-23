#!/usr/bin/env bash
# ============================================================
# 录制 Trocar 装配 episode 数据（独立脚本，不走 RLinf）
# ============================================================
set -e

CONDA_DIR="/localhome/local-vennw/miniconda3"
ISAACLAB_DIR="/localhome/local-vennw/code/IsaacLab"
ISAACSIM_DIR="/localhome/local-vennw/isaac-sim-standalone-6.0.0-rc.22"
MODEL_PATH="/localhome/local-vennw/models/orca_rlinf_weights/rlinf/actor/model_state_dict"
OUTPUT_DIR="/localhome/local-vennw/data/trocar_recorded_1_steps"
OPEN_LOOP_STEPS=1

# ---- 激活 conda 环境 ----
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate isaaclab_develop_6.0

# ---- IsaacSim 环境变量 ----
export ISAAC_PATH="$ISAACSIM_DIR"
export EXP_PATH="$ISAAC_PATH/apps"
export CARB_APP_PATH="$ISAAC_PATH/kit"

# ---- CUDA ----
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# ---- Headless 渲染 ----
unset DISPLAY
export MESA_GL_VERSION_OVERRIDE=4.6

cd "$ISAACLAB_DIR"

./isaaclab.sh -p scripts/tools/record_trocar_episodes.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_episodes 100 \
  --max_steps 300 \
  --denoising_steps 4 \
  --open_loop_steps "$OPEN_LOOP_STEPS" \
  "$@"
