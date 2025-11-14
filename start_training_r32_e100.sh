#!/bin/bash
# Fine-tuning 配置：高品質長訓練版
# Batch size: 8, Gradient Accumulation: 8, Epochs: 100
# LoRA Rank: 32, Total Steps: ~4300

echo "=========================================="
echo "開始 Fine-tuning Encoder"
echo "配置：Rank-32, 100 Epochs, Batch-8"
echo "預計訓練時間：70-90 分鐘"
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
    --epochs 100 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lr 5e-5 \
    --output-dir ./outputs/optical_lora_r32_e100 \
    --num-workers 4 \
    2>&1 | tee ./outputs/optical_lora_r32_e100/training.log

echo ""
echo "=========================================="
echo "訓練完成！"
echo "Checkpoint 位置: ./outputs/optical_lora_r32_e100/"
echo "Log 檔案: ./outputs/optical_lora_r32_e100/training.log"
echo "=========================================="
