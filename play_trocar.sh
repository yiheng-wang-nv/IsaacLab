#!/usr/bin/env bash
# ============================================================
# RLinf 评估脚本 - Assemble Trocar 任务
# ============================================================
set -e

CONDA_DIR="/localhome/local-vennw/miniconda3"
ISAACLAB_DIR="/localhome/local-vennw/code/IsaacLab"
ISAACSIM_DIR="/localhome/local-vennw/isaac-sim-standalone-6.0.0-rc.22"
MODEL_PATH="/localhome/local-vennw/models/orca_rlinf_weights/rlinf/actor/model_state_dict"
CONFIG_PATH="$ISAACLAB_DIR/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config"
CONFIG_NAME="isaaclab_ppo_gr00t_assemble_trocar"

# ---- 激活 conda 环境 ----
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate isaaclab_develop_6.0

# ---- IsaacSim 环境变量 ----
export ISAAC_PATH="$ISAACSIM_DIR"
export EXP_PATH="$ISAAC_PATH/apps"
export CARB_APP_PATH="$ISAAC_PATH/kit"

# ---- CUDA 12.8（flash-attn 运行时需要）----
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# ---- NCCL 稳定性 ----
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# ---- Headless 渲染：强制 EGL，避免 GLXBadFBConfig crash ----
unset DISPLAY
export MESA_GL_VERSION_OVERRIDE=4.6

cd "$ISAACLAB_DIR"

python scripts/reinforcement_learning/rlinf/play.py \
  --config_path "$CONFIG_PATH" \
  --config_name "$CONFIG_NAME" \
  --model_path "$MODEL_PATH" \
  --num_envs 32 \
  --video \
  "$@" \
  2>&1 | tee play.log
