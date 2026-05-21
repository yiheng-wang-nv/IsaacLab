cd /home/nvidia/workspace/yiheng/cosmos-reason2
source .venv/bin/activate

vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path /home/nvidia/workspace/yiheng/IsaacLab \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.60 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
