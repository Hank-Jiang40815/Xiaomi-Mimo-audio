#!/bin/bash
# 使用 V1 (原始版本) 進行訓練 - 在 tmux 中執行

SESSION_NAME="finetune_v1"

echo "=========================================="
echo "🚀 Fine-tune Encoder V1 (Feature Matching)"
echo "=========================================="
echo ""

# 檢查 tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION_NAME' already exists"
    echo "   1. tmux attach -t $SESSION_NAME"
    echo "   2. tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# 配置
OUTPUT_DIR="outputs/optical_lora_v1_r64_e100"
RANK=64
ALPHA=128

echo "配置："
echo "  🎯 LoRA: rank=$RANK, alpha=$ALPHA (2x原本)"
echo "  📦 Batch: 8 × 8 = 64"
echo "  📚 Epochs: 100"
echo "  📈 LR: 5e-5"
echo "  💾 Output: $OUTPUT_DIR"
echo ""
echo "🚀 Starting in tmux..."

tmux new-session -d -s "$SESSION_NAME" bash -c "
    echo '=========================================='
    echo '🐳 Docker 訓練啟動'
    echo '=========================================='
    echo ''
    
    docker run --gpus all --rm \
        -v \"\$(pwd)\":/workspace -w /workspace \
        mimo-audio:latest python finetune_encoder.py \
        --data-dir ./data/splits/finetune_optical \
        --tokenizer-path ./models/MiMo-Audio-Tokenizer \
        --output-dir ./$OUTPUT_DIR \
        --rank $RANK \
        --alpha $ALPHA \
        --batch-size 8 \
        --gradient-accumulation 8 \
        --epochs 100 \
        --lr 5e-5 \
        --save-every 10
    
    EXIT_CODE=\$?
    
    echo ''
    echo '=========================================='
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '✅ 訓練完成！'
    else
        echo '❌ 訓練失敗 (Exit Code: '\$EXIT_CODE')'
    fi
    echo '=========================================='
    echo ''
    echo '按 Enter 繼續...'
    read
"

echo "✅ tmux session '$SESSION_NAME' 已啟動"
echo ""
echo "連接指令："
echo "  tmux attach -t $SESSION_NAME"
