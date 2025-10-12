# MiMo-Audio GPU-enabled runtime (Python 3.12, CUDA 12.1)
# Base: NVIDIA CUDA runtime with cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: Python 3.12 (via deadsnakes), audio utils, git, build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl git \
      ffmpeg libsndfile1 \
      build-essential && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip for Python 3.12
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# Copy project
COPY . /app

# Install CUDA-enabled PyTorch 2.6.0 and Torchaudio 2.6.0 from PyPI (includes CUDA support)
# Then install project requirements (excluding torch/torchaudio) and flash-attn precompiled wheel
RUN python3.12 -m pip install --upgrade --no-cache-dir \
      torch==2.6.0 torchaudio==2.6.0 && \
    awk '!/^torch==/ && !/^torchaudio==/' requirements.txt > /tmp/requirements.no-torch.txt && \
    python3.12 -m pip install --upgrade --no-cache-dir \
      -r /tmp/requirements.no-torch.txt && \
    python3.12 -m pip install --no-cache-dir \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    # Optional but helpful CLI for model downloads used in README
    python3.12 -m pip install --no-cache-dir huggingface-hub

# Default command: run a simple GPU smoke test (no web UI)
CMD ["python3.12", "scripts/smoke_test.py"]
