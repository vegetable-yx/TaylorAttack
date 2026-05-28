#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TAYLOR_ATTACK_IMAGE:-taylorattack:sglang-v0.5.12}"
DOCKER_BIN="${DOCKER_BIN:-sudo docker}"

${DOCKER_BIN} run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size 64g \
  -v "${ROOT_DIR}:/workspace/TaylorMLP" \
  -w /workspace/TaylorMLP \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/workspace/TaylorMLP/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/TaylorMLP/.cache/huggingface \
  "${IMAGE}" "$@"
