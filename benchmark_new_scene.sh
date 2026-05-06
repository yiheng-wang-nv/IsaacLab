source /localhome/local-vennw/code/IsaacLab/omni.sh
# conda activate isaaclab_develop_6.0
# usage: ./benchmark_new_scene.sh [factory|orca|surgical_room] [model_name] [gpu_id]
SCENE=${1:-factory}
MODEL=${2:-sft_4gpu_256bs_50ksteps_success_lt7}
GPU=${3:-1}

CUDA_VISIBLE_DEVICES=$GPU python /localhome/local-vennw/code/IsaacLab/scripts/tools/record_trocar_episodes.py \
  --model_path /localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/$MODEL/checkpoint-50000 \
  --output_dir /localhome/local-vennw/code/benchmark_new_scene/$MODEL/$SCENE \
  --num_episodes 100 \
  --max_steps 512 \
  --use_gr00t_policy \
  --open_loop_steps 8 \
  --fixed_initial_state_dataset /localhome/local-vennw/code/trocar_success_lt_7s_combined \
  --fixed_initial_state_episode 0 \
  --fixed_initial_state_frame 0 \
  --fixed_initial_state_steps 30 \
  --fixed_initial_state_tolerance 0.035 --seed 42 --scene $SCENE
