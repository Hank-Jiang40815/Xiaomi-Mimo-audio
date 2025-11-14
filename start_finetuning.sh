#!/bin/bash
# MiMo-Audio Encoder 微調啟動腳本

set -e

# 配置
TRAIN_SPLIT="data/splits/finetune_optical/train.json"
VAL_SPLIT="data/splits/finetune_optical/val.json"
TOKENIZER_PATH="./models/MiMo-Audio-Tokenizer"
OUTPUT_DIR="./outputs/finetune_encoder"

# 訓練參數
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8  # 有效 batch size = 2 * 8 = 16
EPOCHS=20
LR=1e-4
LORA_RANK=16
LORA_ALPHA=32

# 檢查資料
echo "🔍 Checking data files..."
if [ ! -f "$TRAIN_SPLIT" ]; then
    echo "❌ Training split not found: $TRAIN_SPLIT"
    exit 1
fi

if [ ! -f "$VAL_SPLIT" ]; then
    echo "❌ Validation split not found: $VAL_SPLIT"
    exit 1
fi

echo "✅ Data files found"
echo "   Train: $TRAIN_SPLIT"
echo "   Val: $VAL_SPLIT"

# 檢查模型
echo ""
echo "🔍 Checking model..."
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "❌ Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi
echo "✅ Model found: $TOKENIZER_PATH"

# 顯示配置
echo ""
echo "⚙️  Training Configuration:"
echo "   Batch Size: $BATCH_SIZE"
echo "   Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "   Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "   Epochs: $EPOCHS"
echo "   Learning Rate: $LR"
echo "   LoRA Rank: $LORA_RANK"
echo "   LoRA Alpha: $LORA_ALPHA"
echo "   Output: $OUTPUT_DIR"

# 確認是否繼續
echo ""
read -p "🚀 Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

# 開始訓練
echo ""
echo "🏋️  Starting training..."
echo ""

docker run --gpus all --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    mimo-audio:latest \
    python finetune_encoder.py \
        --train-split "$TRAIN_SPLIT" \
        --val-split "$VAL_SPLIT" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
        --epochs $EPOCHS \
        --lr $LR \
        --lora-rank $LORA_RANK \
        --lora-alpha $LORA_ALPHA \
        --output-dir "$OUTPUT_DIR" \
        --num-workers 4

echo ""
echo "✨ Training completed!"
echo "📁 Check results in: $OUTPUT_DIR"
