#!/usr/bin/env bash
# Launch the checkpoint watcher in a validated vLLM environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VLLM_ENV_PREFIX:-}" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-${VLLM_ENV_PREFIX}/bin/python}
  VLLM_BIN=${VLLM_BIN:-${VLLM_ENV_PREFIX}/bin/vllm}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
  VLLM_BIN=${VLLM_BIN:-vllm}
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] 找不到 Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "[error] 找不到 vLLM: ${VLLM_BIN}" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/async_vllm_eval.py" \
  --vllm-bin "$(command -v "${VLLM_BIN}")" \
  "$@"
