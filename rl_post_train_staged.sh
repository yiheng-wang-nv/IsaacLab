source /localhome/local-vennw/code/IsaacLab/omni.sh
# Staged RLinf post-training: stage 1 -> 2 -> 3 -> 4
# usage: ./rl_post_train_staged.sh [model_name] [gpu_ids] [epochs_per_stage]
MODEL=${1:-sft_4gpu_256bs_50ksteps_success_lt7}
GPUS=${2:-0,1,2,3,4,5,6,7}
EPOCHS_PER_STAGE=${3:-200}

MODEL_PATH=/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/$MODEL/checkpoint-50000
CONFIG_PATH=/localhome/local-vennw/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config
CONFIG_NAME=isaaclab_ppo_gr00t_assemble_trocar
ENV_CFG=/localhome/local-vennw/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/g129_dex3_env_cfg.py
YAML_CFG=$CONFIG_PATH/$CONFIG_NAME.yaml
TRAIN_SCRIPT=/localhome/local-vennw/code/IsaacLab/scripts/reinforcement_learning/rlinf/train.py
LOG_BASE=/localhome/local-vennw/code/IsaacLab/scripts/reinforcement_learning/rlinf/logs/rlinf

set -e

patch_stage() {
    local stage=$1
    local max_steps=$2
    sed -i "s/\"success_stage\": [0-9]*/\"success_stage\": $stage/" $ENV_CFG
    sed -i "s/max_steps_per_rollout_epoch: [0-9]*/max_steps_per_rollout_epoch: $max_steps/" $YAML_CFG
    echo "[staged] success_stage=$stage, max_steps_per_rollout_epoch=$max_steps"
}

find_latest_checkpoint() {
    local log_dir=$1
    # Find highest global_step checkpoint
    ls -d $log_dir/checkpoints/global_step_* 2>/dev/null | \
        sort -t_ -k3 -n | tail -1
}

resume_dir=""

for STAGE in 1 2 3 4; do
    if [ $STAGE -eq 1 ]; then
        MAX_STEPS=128
    else
        MAX_STEPS=64
    fi

    echo "========================================"
    echo "Training Stage $STAGE / 4"
    echo "  max_steps_per_rollout_epoch=$MAX_STEPS"
    echo "  max_epochs=$EPOCHS_PER_STAGE"
    echo "  resume_dir=${resume_dir:-none}"
    echo "========================================"

    patch_stage $STAGE $MAX_STEPS

    # Build command
    CMD="CUDA_VISIBLE_DEVICES=$GPUS python $TRAIN_SCRIPT \
        --config_path $CONFIG_PATH \
        --config_name $CONFIG_NAME \
        --model_path $MODEL_PATH \
        --max_epochs $EPOCHS_PER_STAGE"

    if [ -n "$resume_dir" ]; then
        CMD="$CMD --resume_dir $resume_dir"
    fi

    eval $CMD

    # Find latest checkpoint from this stage's run
    latest_log=$(ls -td $LOG_BASE/*/  2>/dev/null | head -1)
    latest_ckpt=$(find_latest_checkpoint "${latest_log%/}")

    if [ -z "$latest_ckpt" ]; then
        echo "[ERROR] No checkpoint found after stage $STAGE, aborting."
        exit 1
    fi

    resume_dir=$latest_ckpt
    echo "[staged] Stage $STAGE complete. Checkpoint: $resume_dir"
done

echo "========================================"
echo "All 4 stages complete!"
echo "Final checkpoint: $resume_dir"
echo "========================================"
