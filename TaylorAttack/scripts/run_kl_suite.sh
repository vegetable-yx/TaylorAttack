#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TAYLOR_ATTACK_IMAGE:-taylorattack:sglang-v0.5.12}"
DOCKER_BIN="${DOCKER_BIN:-sudo docker}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-32}"
TOP_LOGPROBS="${TOP_LOGPROBS:-32}"
MAX_TOKEN_IDS="${MAX_TOKEN_IDS:-2048}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-8192}"
RUN_SUFFIX="${RUN_SUFFIX:-kl_official_sglang}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

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
    "${env_args[@]}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
    -v "${ROOT_DIR}:/workspace/TaylorMLP" \
    -w /workspace/TaylorMLP \
    "${IMAGE}" \
    bash -lc "export PYTHONPATH=TaylorAttack; ${command}"
}

sample_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  sample_args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<< "${MODELS}"
else
  models=(llama3_8b_instruct gemma4_e2b_it gemma4_e4b_it qwen35_9b qwen36_27b)
fi
declare -A hf_id=(
  [llama3_8b_instruct]=meta-llama/Meta-Llama-3-8B-Instruct
  [gemma4_e2b_it]=google/gemma-4-E2B-it
  [gemma4_e4b_it]=google/gemma-4-E4B-it
  [qwen35_9b]=Qwen/Qwen3.5-9B
  [qwen36_27b]=Qwen/Qwen3.6-27B
)

for model in "${models[@]}"; do
  original="${hf_id[$model]}"
  recovered="TaylorAttack/results/${model}_attack/recovered_model"
  output="TaylorAttack/results/${model}_${RUN_SUFFIX}"
  if [[ ! -d "${ROOT_DIR}/${recovered}" ]]; then
    echo "SKIP ${model}: missing ${recovered}"
    continue
  fi

  model_batch_size="${BATCH_SIZE}"
  model_max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS}"
  model_mem_fraction_static="${SGLANG_MEM_FRACTION_STATIC}"
  model_context_length="${SGLANG_CONTEXT_LENGTH}"
  case "${model}" in
    qwen36_27b)
      model_batch_size="${LARGE_KL_BATCH_SIZE:-8}"
      model_max_running_requests="${LARGE_KL_MAX_RUNNING_REQUESTS:-64}"
      model_mem_fraction_static="${LARGE_KL_MEM_FRACTION_STATIC:-0.78}"
      ;;
  esac
  context_arg=""
  if [[ -n "${model_context_length}" ]]; then
    context_arg="--sglang-context-length ${model_context_length}"
  fi

  echo "===== ${model}: KL backend=sglang output=${output} batch=${model_batch_size} top_k=${TOP_LOGPROBS} context=${model_context_length} ====="
  run_gpu "TaylorAttack/scripts/taylormlp kl \
    --config TaylorAttack/configs/default.yaml \
    --original-model ${original} \
    --recovered-model ${recovered} \
    --model-label ${model} \
    --benchmark math500 \
    --benchmark gpqa_diamond \
    --output ${output} \
    --batch-size ${model_batch_size} \
    --progress-interval ${PROGRESS_INTERVAL} \
    --top-logprobs ${TOP_LOGPROBS} \
    --max-token-ids ${MAX_TOKEN_IDS} \
    --sglang-tp-size ${SGLANG_TP_SIZE} \
    --sglang-mem-fraction-static ${model_mem_fraction_static} \
    --sglang-max-running-requests ${model_max_running_requests} \
    ${context_arg} \
    ${sample_args[*]}" \
    > "${LOG_DIR}/${model}_${RUN_SUFFIX}.log" 2>&1
  echo "KL rc ${model}: $?"
done

python3 - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
run_suffix = "${RUN_SUFFIX}"
rows = []
for directory in sorted((root / "TaylorAttack/results").glob(f"*_{run_suffix}")):
    model = directory.name[: -len(f"_{run_suffix}")]
    for path in sorted(directory.glob("*_kl.json")):
        benchmark = path.name[: -len("_kl.json")]
        values = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": model, "benchmark": benchmark, **values})

out = root / "TaylorAttack/results" / f"{run_suffix}_all.md"
json_out = out.with_suffix(".json")
header = (
    "| Model | Benchmark | Top-K | Tokens | KL mean | KL std | KL median | Run Time |\\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\\n"
)
body = "".join(
    f"| {row['model']} | {row['benchmark']} | {row.get('top_logprobs_num', '')} | "
    f"{row.get('tokens', '')} | {row['mean']:.6g} | {row['std']:.6g} | "
    f"{row['median']:.6g} | {float(row.get('elapsed_sec', 0.0)):.2f}s |\\n"
    for row in rows
)
out.write_text(header + body, encoding="utf-8")
json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"wrote combined KL table to {out}")
PY
