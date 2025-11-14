#!/bin/bash
# 啟動 V2: Codebook-Aware Loss Training

SESSION_NAME="finetune_v2"

echo "=========================================="
echo "🚀 Fine-tune Encoder V2 (Codebook-Aware)"
echo "=========================================="
echo ""
echo "核心改進："
echo "  ✅ Feature Matching Loss"
echo "  ✅ Codebook Alignment Loss (確保 codes 一致)"
echo "  ✅ 直接優化 codebook 映射"
echo ""

# 檢查 tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION_NAME' already exists"
    echo "   1. tmux attach -t $SESSION_NAME"
    echo "   2. tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# 配置
OUTPUT_DIR="outputs/optical_lora_v2_r32_e100"
RANK=32
ALPHA=64

echo "配置："
echo "  🎯 LoRA: rank=$RANK, alpha=$ALPHA"
echo "  📦 Batch: 8 × 8 = 64"
echo "  📚 Epochs: 100"
echo "  📈 LR: 5e-5"
echo "  💾 Output: $OUTPUT_DIR"
echo ""
echo "🚀 Starting in tmux..."

tmux new-session -d -s "$SESSION_NAME" bash -c "
    echo '=========================================='
    echo '🐳 Docker 訓練啟動 (V2 - Codebook-Aware)'
    echo '=========================================='
    echo ''
    
    docker run --gpus all --rm \
        -v \"\$(pwd)\":/workspace -w /workspace \
        mimo-audio:latest python finetune_encoder_v2.py \
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
echo "連接查看："
echo "  tmux attach -t $SESSION_NAME"
