FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libaio-dev \
        libnuma-dev \
        numactl \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/qwen-sft

ENV PATH="/opt/qwen-sft/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    RUN_BACKGROUND=0 \
    MODEL_PATH=/model \
    OUTPUT_ROOT=/output

WORKDIR /app

# Install torch first because flash-attn imports torch while its wheel is built.
RUN python -m pip install --no-cache-dir \
        pip==26.1.2 \
        setuptools==81.0.0 \
        wheel==0.47.0

RUN python -m pip install --no-cache-dir torch==2.11.0

# flash-attn and causal-conv1d are built later in this image. Install Ninja
# before their pip transactions so wheel compilation is actually parallel.
RUN python -m pip install --no-cache-dir ninja==1.13.0

ARG FLASH_ATTN_CUDA_ARCHS=90
ENV FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}"

COPY requirements.lock /tmp/requirements.lock

RUN DS_BUILD_CPU_ADAM=1 MAX_JOBS="${MAX_JOBS}" \
    python -m pip install --no-cache-dir --no-build-isolation \
        -r /tmp/requirements.lock

COPY requirements-fla.lock /tmp/requirements-fla.lock

# Transformers' Qwen3.5 GatedDeltaNet needs both FLA and causal-conv1d.
# PyTorch 2.11 pins Triton 3.6.0, whose Hopper gated-delta backward is blocked
# by FLA for correctness; TileLang supplies the verified backward backend.
# --no-deps guarantees that this add-on cannot replace torch or triton.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        -r /tmp/requirements-fla.lock

RUN python -m pip check \
    && python -c "import importlib.metadata as m, importlib.util; import accelerate, causal_conv1d, datasets, deepspeed, flash_attn, torch, transformers; assert importlib.util.find_spec('fla') is not None; assert importlib.util.find_spec('tilelang') is not None; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, 'deepspeed': deepspeed.__version__, 'datasets': datasets.__version__, 'accelerate': accelerate.__version__, 'flash_attn': flash_attn.__version__, 'flash_linear_attention': m.version('flash-linear-attention'), 'causal_conv1d': m.version('causal-conv1d'), 'tilelang': m.version('tilelang')})"

# Put runtime-only cache settings after dependency installation so changing
# cache layout does not invalidate the expensive PyTorch/CUDA build layers.
ENV TILELANG_CACHE_DIR=/cache/tilelang \
    TRITON_CACHE_DIR=/cache/triton

COPY train/ /app/train/
COPY eval/ /app/eval/

RUN chmod +x /app/train/run_qwen36_19b_a3b_sft_deepspeed.sh \
        /app/train/verify_qwen35_fast_path.py \
        /app/eval/run_async_vllm_eval.sh \
    && mkdir -p \
        /cache/huggingface \
        /cache/torch_extensions \
        /cache/tilelang \
        /cache/triton \
        /model \
        /data \
        /output

ENTRYPOINT ["bash", "/app/train/run_qwen36_19b_a3b_sft_deepspeed.sh"]
