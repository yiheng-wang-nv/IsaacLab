#!/usr/bin/env bash
# Push the local scripts/mhs tree to the simulation machine without going through git.
# Use this for fast iteration; use a git push/pull for checkpoints worth keeping.
#
#   ./scripts/mhs/sync.sh            # local -> remote
set -euo pipefail

REMOTE="${MHS_REMOTE:-nvidia@10.19.224.59}"
REMOTE_REPO="${MHS_REMOTE_REPO:-/home/nvidia/workspace/yiheng/IsaacLab}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -az --delete \
    --exclude '__pycache__' \
    "${LOCAL_REPO}/scripts/mhs/" \
    "${REMOTE}:${REMOTE_REPO}/scripts/mhs/"
echo "synced ${LOCAL_REPO}/scripts/mhs -> ${REMOTE}:${REMOTE_REPO}/scripts/mhs"
