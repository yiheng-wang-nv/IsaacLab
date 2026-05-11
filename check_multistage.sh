DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority \
  ./isaaclab.sh -p scripts/tools/interactive_trocar_task_complete.py \
    --model_path /home/nvidia/workspace/yiheng/models/sim6_gr00t_n15_100ksteps_split_stage \
    --gr00t_root /home/nvidia/workspace/yiheng/IsaacLab/Isaac-GR00T \
    --progress_regressor_path /home/nvidia/workspace/yiheng/models/task_progress_regressor_ordered/best_model.pt \
    --device cuda:0 \
    --open_loop_steps 8 \
    --fixed_initial_state_dataset /home/nvidia/workspace/yiheng/datasets/trocar_success_lt_7s_split_by_stage_task_complete \
    --fixed_initial_state_episode 0 \
    --fixed_initial_state_frame 0 \
    --fixed_initial_state_steps 10 \
    --fixed_initial_state_tolerance 0.035 --seed 42 --task_complete_threshold 0.9 \
    --stage_precondition_threshold 0.90 \
    --task_complete_peak_threshold 0.90 \
    --task_complete_drop_threshold 0.20
