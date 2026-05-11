./isaaclab.sh -p scripts/tools/extract_task_complete_features.py \
  --checkpoint /home/nvidia/workspace/yiheng/models/sim6_gr00t_n15_100ksteps_split_stage \
  --gr00t_root /home/nvidia/workspace/yiheng/IsaacLab/Isaac-GR00T \
  --dataset /home/nvidia/workspace/yiheng/datasets/trocar_success_lt_7s_split_by_stage_task_complete \
  --output_dir /home/nvidia/workspace/yiheng/datasets/task_complete_features_cross \
  --device cuda:0 \
  --batch_size 16 \
  --cross_task_negatives \
  --neighbor_exclusion_frac 0.10
