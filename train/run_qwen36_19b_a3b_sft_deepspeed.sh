#!/usr/bin/env bash
# Portable single-node/multi-node DeepSpeed launcher for Qwen3.6-19B-A3B SFT.
#
# Required:
#   MODEL_PATH=/path/to/model
#   --data-files /path/to/train-a.jsonl [/path/to/train-b.jsonl ...]
#
# Optional validation:
#   --eval-data-files /path/to/val.jsonl --eval-steps 70
#
# Local environment:
#   MODEL_PATH=/path/to/model OUTPUT_DIR=/path/to/output \
#     bash train/run_qwen36_19b_a3b_sft_deepspeed.sh \
#     --gpus 0,1 --data-files /path/to/train.jsonl
#
# Docker convention:
#   MODEL_PATH=/model OUTPUT_DIR=/output/run-1 \
#     bash train/run_qwen36_19b_a3b_sft_deepspeed.sh \
#     --gpus 0,1 --data-files /data/train.jsonl
#
# Multi-node convention (run once per node with a unique NODE_RANK):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29508 \
#   MODEL_PATH=/model OUTPUT_DIR=/output/run-1 \
#     bash train/run_qwen36_19b_a3b_sft_deepspeed.sh \
#     --gpus 0,1,2,3,4,5,6,7 --data-files /data/train.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ "${RUN_BACKGROUND:-0}" == "1" && -z "${RUN_BACKGROUND_CHILD:-}" ]]; then
  LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT:-${REPO_ROOT}/logs}}"
  mkdir -p "${LOG_DIR}"
  RUN_LOG_FILE="${RUN_LOG_FILE:-${LOG_DIR}/qwen36_19b_a3b_sft_$(date +%Y%m%d_%H%M%S).log}"
  nohup env RUN_BACKGROUND=0 RUN_BACKGROUND_CHILD=1 RUN_LOG_FILE="${RUN_LOG_FILE}" \
    bash "${SCRIPT_PATH}" "$@" >"${RUN_LOG_FILE}" 2>&1 &
  echo "[launch] PID=$!"
  echo "[launch] LOG=${RUN_LOG_FILE}"
  exit 0
fi

cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-/model}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/ds_config/qwen36_19b_a3b_zero3.json}
EXPECTED_NUM_EXPERTS=${EXPECTED_NUM_EXPERTS:-128}

if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}
  TORCHRUN_BIN=${TORCHRUN_BIN:-${CONDA_ENV_PREFIX}/bin/torchrun}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
  TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] 找不到 Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  echo "[error] 找不到 torchrun: ${TORCHRUN_BIN}" >&2
  exit 2
fi

# DeepSpeedCPUAdam JIT-loads a C++ extension through ninja. Prefer tools from
# the same Python environment over user-level executables.
PYTHON_DIR="$(cd "$(dirname "$(command -v "${PYTHON_BIN}")")" && pwd)"
export PATH="${PYTHON_DIR}:${PATH}"

for argument in "$@"; do
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    echo "Launcher option: --gpus 0,1,... (default: CUDA_VISIBLE_DEVICES or 0)"
    echo "Multi-node env: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT"
    echo "  NNODES=1 (default) uses standalone mode."
    echo "  NNODES>1 requires one unique NODE_RANK per node and a shared MASTER_ADDR."
    exec "${PYTHON_BIN}" train/train_qwen36_19b_a3b_sft_deepspeed.py --help
  fi
done

GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
TRAIN_ARGS=()
HAS_DATA_FILES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -lt 2 ]] && { echo "[error] --gpus 需要逗号分隔的 GPU 列表" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#--gpus=}"
      shift
      ;;
    --data-files|--data-files=*)
      HAS_DATA_FILES=1
      TRAIN_ARGS+=("$1")
      shift
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done
export CUDA_VISIBLE_DEVICES="${GPUS}"

if [[ -z "${GPUS}" ]]; then
  echo "[error] GPU 列表不能为空" >&2
  exit 2
fi
if (( HAS_DATA_FILES == 0 )); then
  echo "[error] 必须通过 --data-files 提供至少一个本地训练数据文件" >&2
  echo "[example] --data-files /data/train.jsonl" >&2
  exit 2
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
  NPROC_PER_NODE=${#GPU_LIST[@]}
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NPROC_PER_NODE 必须大于 0" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

USE_LORA=${USE_LORA:-false}
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-true}
LAST_ASSISTANT_ONLY=${LAST_ASSISTANT_ONLY:-false}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
EVAL_STEPS=${EVAL_STEPS:-70}
EVAL_METRIC=${EVAL_METRIC:-loss}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-256}
EVAL_GENERATION_ENABLE_THINKING=${EVAL_GENERATION_ENABLE_THINKING:-false}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-false}
ASYNC_EVAL_MARKERS=${ASYNC_EVAL_MARKERS:-false}
REPORT_TO=${REPORT_TO:-none}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-29508}
MASTER_ADDR=${MASTER_ADDR:-}
PRECHECK_ONLY=${PRECHECK_ONLY:-0}
SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-0}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]
}

if ! is_positive_integer "${NNODES}"; then
  echo "[error] NNODES 必须是大于 0 的整数，当前值: ${NNODES}" >&2
  exit 2
fi
if ! is_nonnegative_integer "${NODE_RANK}" || (( NODE_RANK >= NNODES )); then
  echo "[error] NODE_RANK 必须是 [0, NNODES) 内的整数，当前值: ${NODE_RANK}" >&2
  exit 2
fi
if ! is_positive_integer "${MASTER_PORT}" || (( MASTER_PORT > 65535 )); then
  echo "[error] MASTER_PORT 必须是 1-65535 内的整数，当前值: ${MASTER_PORT}" >&2
  exit 2
fi
if (( NNODES == 1 && NODE_RANK != 0 )); then
  echo "[error] 单机模式要求 NODE_RANK=0" >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR}" ]]; then
  echo "[error] 多机模式要求设置 MASTER_ADDR（node 0 的可达 IP 或主机名）" >&2
  exit 2
fi

if [[ "${USE_LORA}" == "true" || "${USE_LORA}" == "1" ]]; then
  RUN_PREFIX=qwen36-19b-a3b-lora-sft
else
  RUN_PREFIX=qwen36-19b-a3b-full-sft
fi
RUN_NAME=${RUN_NAME:-${RUN_PREFIX}-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT:-/output}/${RUN_NAME}}

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --expected-num-experts "${EXPECTED_NUM_EXPERTS}"
  --max-samples -1
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --eval-steps "${EVAL_STEPS}"
  --eval-metric "${EVAL_METRIC}"
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --eval-generation-enable-thinking "${EVAL_GENERATION_ENABLE_THINKING}"
  --use-lora "${USE_LORA}"
  --freeze-vision-tower "${FREEZE_VISION_TOWER}"
  --last-assistant-only "${LAST_ASSISTANT_ONLY}"
  --save-only-model "${SAVE_ONLY_MODEL}"
  --async-eval-markers "${ASYNC_EVAL_MARKERS}"
  --report-to "${REPORT_TO}"
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
)

resolve_cli_value() {
  local option="$1"
  local value="$2"
  local argument_index
  for ((argument_index = 0; argument_index < ${#TRAIN_ARGS[@]}; argument_index++)); do
    if [[ "${TRAIN_ARGS[argument_index]}" == "${option}" ]]; then
      if (( argument_index + 1 < ${#TRAIN_ARGS[@]} )); then
        value="${TRAIN_ARGS[argument_index + 1]}"
      fi
    elif [[ "${TRAIN_ARGS[argument_index]}" == "${option}="* ]]; then
      value="${TRAIN_ARGS[argument_index]#*=}"
    fi
  done
  printf '%s' "${value}"
}

EFFECTIVE_EVAL_STEPS="$(resolve_cli_value --eval-steps "${EVAL_STEPS}")"
EFFECTIVE_EVAL_METRIC="$(resolve_cli_value --eval-metric "${EVAL_METRIC}")"
EFFECTIVE_EVAL_MAX_NEW_TOKENS="$(
  resolve_cli_value --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
)"
EFFECTIVE_EVAL_GENERATION_ENABLE_THINKING="$(
  resolve_cli_value \
    --eval-generation-enable-thinking "${EVAL_GENERATION_ENABLE_THINKING}"
)"
EFFECTIVE_ASYNC_EVAL_MARKERS="$(
  resolve_cli_value --async-eval-markers "${ASYNC_EVAL_MARKERS}"
)"

echo "PYTHON_BIN=$(command -v "${PYTHON_BIN}")"
echo "TORCHRUN_BIN=$(command -v "${TORCHRUN_BIN}")"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORLD_SIZE=$((NNODES * NPROC_PER_NODE))"
echo "MASTER_ADDR=${MASTER_ADDR:-standalone}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "USE_LORA=${USE_LORA}"
echo "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER}"
echo "LAST_ASSISTANT_ONLY=${LAST_ASSISTANT_ONLY}"
echo "MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH}"
echo "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}"
echo "EVAL_STEPS=${EFFECTIVE_EVAL_STEPS}"
echo "EVAL_METRIC=${EFFECTIVE_EVAL_METRIC}"
echo "EVAL_MAX_NEW_TOKENS=${EFFECTIVE_EVAL_MAX_NEW_TOKENS}"
echo "EVAL_GENERATION_ENABLE_THINKING=${EFFECTIVE_EVAL_GENERATION_ENABLE_THINKING}"
echo "ASYNC_EVAL_MARKERS=${EFFECTIVE_ASYNC_EVAL_MARKERS}"

SKIP_GPU_BOOL=false
if [[ "${SKIP_GPU_CHECK}" == "1" ]]; then
  SKIP_GPU_BOOL=true
fi

set +e
"${PYTHON_BIN}" train/train_qwen36_19b_a3b_sft_deepspeed.py \
  "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" \
  --preflight-only true --skip-gpu-check "${SKIP_GPU_BOOL}"
PREFLIGHT_STATUS=$?
set -e
if (( PREFLIGHT_STATUS != 0 )); then
  echo "[hint] 请确认模型/数据路径正确，并检查当前 Python 环境中的训练依赖。" >&2
  exit "${PREFLIGHT_STATUS}"
fi

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[info] PRECHECK_ONLY=1；预检通过，未启动训练。"
  exit 0
fi

if [[ -n "${RUN_LOG_FILE:-}" ]]; then
  echo "[log] ${RUN_LOG_FILE}"
fi

TORCHRUN_DISTRIBUTED_ARGS=(
  --nnodes "${NNODES}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
)
if (( NNODES == 1 )); then
  TORCHRUN_DISTRIBUTED_ARGS=(--standalone "${TORCHRUN_DISTRIBUTED_ARGS[@]}")
else
  TORCHRUN_DISTRIBUTED_ARGS+=(
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
  )
fi

"${TORCHRUN_BIN}" \
  "${TORCHRUN_DISTRIBUTED_ARGS[@]}" \
  train/train_qwen36_19b_a3b_sft_deepspeed.py \
  "${COMMON_ARGS[@]}" \
  "${TRAIN_ARGS[@]}"
