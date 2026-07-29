#!/usr/bin/env bash
# Start decoupled SFT training and checkpoint-based vLLM pass-rate evaluation.
set -euo pipefail

TRAIN_IMAGE=${TRAIN_IMAGE:-qwen36-sft:1.0}
EVAL_IMAGE=${EVAL_IMAGE:-qwen36-vllm-eval:1.0}
MODEL_DIR=${MODEL_DIR:-/mnt/data/user01/qwen3-19b-a3b}
DATA_DIR=${DATA_DIR:-/mnt/data/user01/LLMData}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/data/user01/qwen36_19b_a3b_sft}
LOG_DIR=${LOG_DIR:-/home/user01/jx/opdForqwen/logs}
TRAIN_GPUS=${TRAIN_GPUS:-1,2,3,4,6}
EVAL_GPUS=${EVAL_GPUS:-0,7}
RUN_NAME=${RUN_NAME:-qwen36-19b-a3b-async-test-$(date +%Y%m%d_%H%M%S)}

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.train.log"
EVAL_LOG="${LOG_DIR}/${RUN_NAME}.async-eval.log"
TRAIN_CONTAINER="qwen36-train-${RUN_NAME}"
EVAL_CONTAINER="qwen36-eval-${RUN_NAME}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] 文件不存在: ${path}" >&2
    exit 2
  fi
}

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[error] 模型目录无 config.json: ${MODEL_DIR}" >&2
  exit 2
fi

TRAIN_FILES=(
  split/Nemotron-SFT-Math-v4/train25w.jsonl
  split/OpenMathReasoning/train20w.jsonl
  split/OpenR1-Math-220k/train.jsonl
  split/OpenCodeReasoning-2/train20w.jsonl
  split/general/qwen3_235b_thinking_2507_110k_sft.jsonl
  split/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl
)
for relative_path in "${TRAIN_FILES[@]}"; do
  require_file "${DATA_DIR}/${relative_path}"
done
require_file "${DATA_DIR}/val/HARP/HARP_difficulty_2_sample_50.jsonl"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if docker container inspect "${TRAIN_CONTAINER}" >/dev/null 2>&1; then
  echo "[error] 容器名已存在: ${TRAIN_CONTAINER}" >&2
  exit 2
fi
if docker container inspect "${EVAL_CONTAINER}" >/dev/null 2>&1; then
  echo "[error] 容器名已存在: ${EVAL_CONTAINER}" >&2
  exit 2
fi

if ! docker image inspect "${TRAIN_IMAGE}" >/dev/null 2>&1; then
  echo "[error] 找不到训练镜像: ${TRAIN_IMAGE}" >&2
  exit 2
fi
if ! docker image inspect "${EVAL_IMAGE}" >/dev/null 2>&1; then
  echo "[error] 找不到验证镜像: ${EVAL_IMAGE}" >&2
  exit 2
fi

nohup docker run --rm \
  --name "${EVAL_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${OUTPUT_ROOT}:/output" \
  -v qwen36-vllm-cache:/cache \
  "${EVAL_IMAGE}" \
  --run-dir "/output/${RUN_NAME}" \
  --dataset-file /datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl \
  --metadata-source /model \
  --gpus "${EVAL_GPUS}" \
  --tensor-parallel-size 2 \
  --max-model-len 24000 \
  --max-tokens 18000 \
  --thinking true \
  --temperature 0 \
  --concurrency 16 \
  --poll-seconds 15 \
  >"${EVAL_LOG}" 2>&1 &
EVAL_LAUNCH_PID=$!

nohup docker run --rm \
  --name "${TRAIN_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${OUTPUT_ROOT}:/output" \
  -v qwen36-torch-cache:/cache \
  -e MODEL_PATH=/model \
  -e OUTPUT_ROOT=/output \
  -e RUN_NAME="${RUN_NAME}" \
  "${TRAIN_IMAGE}" \
  --gpus "${TRAIN_GPUS}" \
  --data-files \
    /datasets/split/Nemotron-SFT-Math-v4/train25w.jsonl \
    /datasets/split/OpenMathReasoning/train20w.jsonl \
    /datasets/split/OpenR1-Math-220k/train.jsonl \
    /datasets/split/OpenCodeReasoning-2/train20w.jsonl \
    /datasets/split/general/qwen3_235b_thinking_2507_110k_sft.jsonl \
    /datasets/split/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl \
  --max-samples-per-file \
    2000 2000 2000 2000 1400 600 \
  --last-assistant-only-per-file \
    false false false false false true \
  --enable-thinking-per-file \
    true true true true true true \
  --max-seq-length 24000 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --use-lora false \
  --freeze-vision-tower true \
  --save-steps 10 \
  --save-total-limit 10 \
  --save-only-model false \
  --async-eval-markers true \
  --report-to none \
  >"${TRAIN_LOG}" 2>&1 &
TRAIN_LAUNCH_PID=$!

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_CONTAINER=${TRAIN_CONTAINER}"
echo "TRAIN_LAUNCH_PID=${TRAIN_LAUNCH_PID}"
echo "TRAIN_LOG=${TRAIN_LOG}"
echo "EVAL_CONTAINER=${EVAL_CONTAINER}"
echo "EVAL_LAUNCH_PID=${EVAL_LAUNCH_PID}"
echo "EVAL_LOG=${EVAL_LOG}"
echo "RESULTS=${RUN_DIR}/async_eval/results.jsonl"
