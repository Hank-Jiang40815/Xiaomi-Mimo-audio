# MiMo-Audio GPU-enabled runtime with PyTorch 2.7.0 for RTX 5090 support
# Base: PyTorch 2.7.0 with CUDA 12.8 development tools (includes nvcc for flash-attn compilation)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install all dependencies in one layer
RUN pip install --no-cache-dir torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir flash-attn --no-build-isolation && \
    pip install --no-cache-dir -r requirements.txt

# Default command: run audio enhancement experiment
CMD ["python", "experiment_audio_enhancement.py"]
