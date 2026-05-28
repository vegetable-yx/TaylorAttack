#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TAYLOR_ATTACK_IMAGE:-taylorattack:sglang-v0.5.12}"
DOCKER_BIN="${DOCKER_BIN:-sudo docker}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
ARCHIVE_EXISTING="${ARCHIVE_EXISTING:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CALIBRATION_MAX_TOKENS="${CALIBRATION_MAX_TOKENS:-20000}"

mkdir -p "${LOG_DIR}"

_env_args() {
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    echo "--env-file ${ROOT_DIR}/.env"
  else
    echo "-e HF_TOKEN=${HF_TOKEN:-} -e HF_HOME=/workspace/TaylorMLP/.cache/huggingface -e TRANSFORMERS_CACHE=/workspace/TaylorMLP/.cache/huggingface"
  fi
}

run_docker() {
  local command="$1"
  local env_args=()
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    env_args+=(--env-file "${ROOT_DIR}/.env")
  else
    env_args+=(
      -e "HF_TOKEN=${HF_TOKEN:-}"
      -e "HF_HOME=/workspace/TaylorMLP/.cache/huggingface"
      -e "TRANSFORMERS_CACHE=/workspace/TaylorMLP/.cache/huggingface"
    )
  fi
  ${DOCKER_BIN} run --rm \
    --ipc=host \
    --shm-size 64g \
    --network=host \
    "${env_args[@]}" \
    -v "${ROOT_DIR}:/workspace/TaylorMLP" \
    -w /workspace/TaylorMLP \
    "${IMAGE}" \
    bash -lc "export PYTHONPATH=TaylorAttack; ${command}"
}

run_gpu() {
  local command="$1"
  local env_args=()
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    env_args+=(--env-file "${ROOT_DIR}/.env")
  else
    env_args+=(
      -e "HF_TOKEN=${HF_TOKEN:-}"
      -e "HF_HOME=/workspace/TaylorMLP/.cache/huggingface"
      -e "TRANSFORMERS_CACHE=/workspace/TaylorMLP/.cache/huggingface"
    )
  fi
  ${DOCKER_BIN} run --rm --gpus all \
    --ipc=host \
    --shm-size 64g \
    --network=host \
    "${env_args[@]}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -v "${ROOT_DIR}:/workspace/TaylorMLP" \
    -w /workspace/TaylorMLP \
    "${IMAGE}" \
    bash -lc "export PYTHONPATH=TaylorAttack; ${command}"
}

# --- Optional download step ---
if [[ "${DOWNLOAD_FIRST:-1}" == "1" ]]; then
  run_docker "TaylorAttack/scripts/taylormlp download \
    --config TaylorAttack/configs/default.yaml \
    --models \
    --datasets \
    --output TaylorAttack/data"
fi

# --- Calibrate + attack ---
default_models=(llama3_8b_instruct gemma4_e2b_it gemma4_e4b_it qwen35_9b qwen36_27b)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<< "${MODELS}"
else
  models=("${default_models[@]}")
fi

for model in "${models[@]}"; do
  stats="TaylorAttack/results/calibration/${model}_hidden_states.pt"
  output="TaylorAttack/results/${model}_attack"

  if [[ ! -f "${ROOT_DIR}/${stats}" ]]; then
    echo "===== ${model}: calibration output=${stats} ====="
    run_gpu "TaylorAttack/scripts/taylormlp calibrate \
      --config TaylorAttack/configs/default.yaml \
      --model ${model} \
      --output ${stats} \
      --max-tokens ${CALIBRATION_MAX_TOKENS}" \
      > "${LOG_DIR}/${model}_calibrate.log" 2>&1
    echo "calibration rc ${model}: $?"
  fi

  if [[ -d "${ROOT_DIR}/${output}" && "${ARCHIVE_EXISTING}" == "1" ]]; then
    mv "${ROOT_DIR}/${output}" "${ROOT_DIR}/${output}_before_fix_${RUN_STAMP}"
  fi

  echo "===== ${model}: attack output=${output} ====="
  run_gpu "TaylorAttack/scripts/taylormlp attack \
    --config TaylorAttack/configs/default.yaml \
    --model ${model} \
    --stats-file ${stats} \
    --output ${output} \
    --save-recovered-model" \
    > "${LOG_DIR}/${model}_attack.log" 2>&1
  echo "attack rc ${model}: $?"
done

# --- Aggregate results ---
python3 - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
rows = []
for path in sorted((root / "TaylorAttack/results").glob("*_attack/attack_metrics.json")):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not metrics:
        continue
    model = path.parent.name[: -len("_attack")]
    total = sum(int(item["taylormlp_params"]) for item in metrics)
    recovered = sum(int(item["recovered_params"]) for item in metrics)
    elapsed = sum(float(item["elapsed_sec"]) for item in metrics)
    down_p99 = max(
        float(item["categories"]["down_sampled"].get("column_relative_error_p99", 0.0))
        for item in metrics
    )
    down_max = max(
        float(item["categories"]["down_sampled"].get("column_relative_error_max", 0.0))
        for item in metrics
    )
    rows.append({
        "model": model,
        "taylormlp_params": total,
        "recovered_params": recovered,
        "recovered_ratio": recovered / total if total else 1.0,
        "elapsed_sec": elapsed,
        "down_column_p99_max": down_p99,
        "down_column_error_max": down_max,
    })

out = root / "TaylorAttack/results/table1_attack_performance_all.md"
json_out = out.with_suffix(".json")
header = (
    "| Model | # TaylorMLP Parameters | # Recovered Parameters | "
    "Recovered Ratio | Run Time | Max Down Col p99 | Max Down Col Error |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
)
body = "".join(
    f"| {row['model']} | {row['taylormlp_params']:,} | "
    f"{row['recovered_params']:,} | {row['recovered_ratio']:.2%} | "
    f"{row['elapsed_sec']:.2f}s | {row['down_column_p99_max']:.6g} | "
    f"{row['down_column_error_max']:.6g} |\n"
    for row in rows
)
out.write_text(header + body, encoding="utf-8")
json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"wrote combined attack table to {out}")
PY
