#!/usr/bin/env bash
set -euo pipefail

# Sequentially run CodeRefiner experiments with waveform normalization.
# This script只負責「訓練」，不做任何 git 紀錄或 commit。

IMAGE="mimo-audio:latest"

docker_run() {
  local cmd="$1"
  docker run --gpus all --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "${IMAGE}" \
    bash -lc "${cmd}"
}

echo ">>> Starting CodeRefiner WaveNorm sweep (N=8,12,16,20)"

for N in 8 12 16 20; do
  OUT_DIR="outputs/code_refiner_optical_e100_q${N}_norm"
  echo "=== Training CodeRefiner N=${N} (WaveNorm) -> ${OUT_DIR} ==="
  docker_run "python finetune_code_refiner.py \
    --data-dir ./data/splits/finetune_optical \
    --tokenizer-path ./models/MiMo-Audio-Tokenizer \
    --output-dir ${OUT_DIR} \
    --epochs 100 --batch-size 4 --lr 1e-4 \
    --num-quantizer-layers ${N} \
    --normalize-waveform"
done

echo '>>> CodeRefiner WaveNorm sweep finished.'

