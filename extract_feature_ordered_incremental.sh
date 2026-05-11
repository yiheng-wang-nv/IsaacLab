#!/usr/bin/env bash
set -euo pipefail

source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_develop_6.0

python scripts/tools/augment_ordered_task_progress_features.py \
  --existing_dir /home/nvidia/workspace/yiheng/datasets/task_complete_features_cross \
  --output_dir /home/nvidia/workspace/yiheng/datasets/task_complete_features_ordered \
  --checkpoint /home/nvidia/workspace/yiheng/models/sim6_gr00t_n15_100ksteps_split_stage \
  --gr00t_root /home/nvidia/workspace/yiheng/IsaacLab/Isaac-GR00T \
  --dataset /home/nvidia/workspace/yiheng/datasets/trocar_success_lt_7s_split_by_stage_task_complete \
  --device cuda:0
