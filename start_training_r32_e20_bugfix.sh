#!/bin/bash
# LoRA Rank-32, 20-epoch 快速驗證訓練（Bug 修復後）
# 用於驗證 audio_lens 修復是否正確

echo "=========================================="
echo "開始 Bug 修復驗證訓練"
echo "配置：Rank-32, 20 Epochs, Batch-8"
echo "預計訓練時間：10-15 分鐘"
echo "=========================================="
echo ""

# 檢查 GPU
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

echo ""
echo "開始訓練..."
echo ""

docker run --gpus all --rm \
    -v "$(pwd)":/workspace -w /workspace \
    mimo-audio:latest python finetune_encoder.py \
    --train-split data/splits/finetune_optical/train.json \
    --val-split data/splits/finetune_optical/val.json \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --epochs 20 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lr 5e-5 \
    --output-dir ./outputs/optical_lora_r32_e20_bugfix \
    --num-workers 4 \
    2>&1 | tee ./outputs/optical_lora_r32_e20_bugfix/training.log

echo ""
echo "=========================================="
echo "驗證訓練完成！"
echo "Checkpoint 位置: ./outputs/optical_lora_r32_e20_bugfix/"
echo "Log 檔案: ./outputs/optical_lora_r32_e20_bugfix/training.log"
echo "=========================================="
