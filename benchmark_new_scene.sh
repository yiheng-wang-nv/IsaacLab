source /localhome/local-vennw/code/IsaacLab/omni.sh
# conda activate isaaclab_develop_6.0
# usage: ./benchmark_new_scene.sh [scene] [policy_type] [model_path_or_name] [gpu_id] [chunk] [init_pose]
#   scene:       lightwheel|factory|orca|surgical_room   (lightwheel = original scene03.usd)
#   policy_type: gr00t (SFT) | rlinf
#   model:       for gr00t: model name under cosmos_gr00t/Isaac-GR00T/<name>/checkpoint-50000
#                for rlinf: absolute path to actor/model_state_dict directory
#   chunk:       open_loop_steps (default 8). Use 1 to match data-collection setting.
#   init_pose:   dataset (default) -- warm-start to episode 0 frame 0 pose from the combined dataset
#                rlinf            -- skip warm-start, use env's default reset pose (matches RLinf training distribution)
#
# Examples:
#   ./benchmark_new_scene.sh lightwheel gr00t sft_4gpu_256bs_50ksteps_success_lt7 1
#   ./benchmark_new_scene.sh factory rlinf <ckpt> 1 1 rlinf

SCENE=${1:-factory}
POLICY=${2:-gr00t}
MODEL=${3:-sft_4gpu_256bs_50ksteps_success_lt7}
GPU=${4:-1}
CHUNK=${5:-8}
INIT_POSE=${6:-dataset}

if [ "$POLICY" = "gr00t" ]; then
    MODEL_PATH=/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/$MODEL/checkpoint-50000
    POLICY_FLAG="--use_gr00t_policy"
    OUT_NAME=$MODEL
elif [ "$POLICY" = "rlinf" ]; then
    MODEL_PATH=$MODEL  # absolute path
    POLICY_FLAG=""
    # derive a short name from the path: parent_of_actor (e.g. global_step_166)
    OUT_NAME=$(basename $(dirname $(dirname "$MODEL_PATH")))_rlinf
else
    echo "Unknown policy_type: $POLICY (use 'gr00t' or 'rlinf')"
    exit 1
fi

if [ "$INIT_POSE" = "dataset" ]; then
    INIT_FLAGS="--fixed_initial_state_dataset /localhome/local-vennw/code/trocar_success_lt_7s_combined \
        --fixed_initial_state_episode 0 \
        --fixed_initial_state_frame 0 \
        --fixed_initial_state_steps 30 \
        --fixed_initial_state_tolerance 0.035"
    POSE_SUFFIX=""
elif [ "$INIT_POSE" = "rlinf" ]; then
    INIT_FLAGS=""
    POSE_SUFFIX="_rlinfpose"
else
    echo "Unknown init_pose: $INIT_POSE (use 'dataset' or 'rlinf')"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path "$MODEL_PATH" \
  --output_dir /localhome/local-vennw/code/benchmark_new_scene/$OUT_NAME/${SCENE}_chunk${CHUNK}${POSE_SUFFIX} \
  --num_episodes 100 \
  --max_steps 512 \
  $POLICY_FLAG \
  --open_loop_steps $CHUNK \
  $INIT_FLAGS \
  --seed 42 --scene $SCENE
