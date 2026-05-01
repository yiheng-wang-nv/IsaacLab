#!/usr/bin/env bash
# ============================================================
# 录制 Trocar 装配 episode 数据（独立脚本）
# ============================================================
set -e

CONDA_DIR="/localhome/local-vennw/miniconda3"
ISAACLAB_DIR="/localhome/local-vennw/code/IsaacLab"
ISAACSIM_DIR="/localhome/local-vennw/isaac-sim-standalone-6.0.0-rc.22"
MODEL_PATH="/localhome/local-vennw/models/orca_rlinf_weights/rlinf/actor/model_state_dict"
OUTPUT_DIR="/localhome/local-vennw/code/trocar_recorded_1_steps_default_rand_light_debug"
OPEN_LOOP_STEPS=1
NUM_EPISODES=100
MAX_STEPS=300

# ---- 数据记录裁剪 ----
# 使用环境 default reset；跳过前几帧，避免记录刚 reset 后的轻微 settle。
SKIP_FIRST_N=3
SKIP_LAST_N=1

# ---- 随机化 ----
TRAY_YAW_MIN_DEG=0
TRAY_YAW_MAX_DEG=10

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
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --denoising_steps 4 \
  --open_loop_steps "$OPEN_LOOP_STEPS" \
  --skip_first_n "$SKIP_FIRST_N" \
  --skip_last_n "$SKIP_LAST_N" \
  --tray_yaw_min_deg "$TRAY_YAW_MIN_DEG" \
  --tray_yaw_max_deg "$TRAY_YAW_MAX_DEG" \
  --randomize_lighting \
  "$@"
