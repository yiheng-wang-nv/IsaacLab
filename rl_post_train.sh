source /localhome/local-vennw/code/IsaacLab/omni.sh
# conda activate isaaclab_develop_6.0
# usage: ./rl_post_train.sh [model_name] [gpu_ids] [num_envs] [max_epochs]
MODEL=${1:-sft_4gpu_256bs_50ksteps_success_lt7}
GPUS=${2:-0,1,2,3,4,5,6,7}
NUM_ENVS=${3:-64}
MAX_EPOCHS=${4:-1000}

MODEL_PATH=/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/$MODEL/checkpoint-50000
CONFIG_PATH=/localhome/local-vennw/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config

LOG_DIR=/localhome/local-vennw/code/IsaacLab/train_logs
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/rlinf_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=$GPUS python -u /localhome/local-vennw/code/IsaacLab/scripts/reinforcement_learning/rlinf/train.py \
  --config_path $CONFIG_PATH \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path $MODEL_PATH \
  --num_envs $NUM_ENVS \
  --max_epochs $MAX_EPOCHS \
  2>&1 | grep -v -E "(Non-GPU-compatible convex mesh|deformable volume tetrahedron|PxgNphaseImplementationContext)" \
       | tee "$LOG_FILE"
