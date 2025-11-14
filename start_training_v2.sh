#!/bin/bash
# 啟動 Encoder Fine-tuning V2 (Codebook-Aware)
# 改進策略：加入 Codebook Alignment Loss

SESSION_NAME="finetune_v2"

echo "=========================================="
echo "🚀 Fine-tune Encoder V2 (Codebook-Aware)"
echo "=========================================="
echo ""
echo "改進項目："
echo "  ✅ Feature Matching Loss (MSE)"
echo "  ✅ Codebook Alignment Loss (L1)"
echo "  ✅ 確保 noisy→clean 映射到相同 codebook"
echo ""

# 配置
DATA_DIR="./data/splits/finetune_optical"
TOKENIZER="./models/MiMo-Audio-Tokenizer"
OUTPUT_DIR="./outputs/optical_lora_v2_r32_e100"
RANK=32
ALPHA=64
BATCH_SIZE=8
GRAD_ACCUM=8
EPOCHS=100
LR=5e-5

echo "配置："
echo "  📁 Data: $DATA_DIR"
echo "  🎯 LoRA: rank=$RANK, alpha=$ALPHA"
echo "  📦 Batch: $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  📚 Epochs: $EPOCHS"
echo "  📈 LR: $LR"
echo "  💾 Output: $OUTPUT_DIR"
echo ""

# 檢查必要檔案
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$TOKENIZER" ]; then
    echo "❌ Tokenizer not found: $TOKENIZER"
    exit 1
fi

# 檢查 tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION_NAME' already exists"
    echo "   1. tmux attach -t $SESSION_NAME  # 連接"
    echo "   2. tmux kill-session -t $SESSION_NAME  # 刪除後重新執行"
    exit 1
fi

echo "🚀 Starting training in tmux..."
echo ""
echo "指令："
echo "  連接: tmux attach -t $SESSION_NAME"
echo "  分離: Ctrl+B 然後按 D"
echo ""

# 在 tmux 中執行
tmux new-session -d -s "$SESSION_NAME" bash -c "
    set -e
    
    echo '=========================================='
    echo '🐳 Running in Docker Container'
    echo '=========================================='
    echo ''
    
    docker run --gpus all --rm \
        -v \"\$(pwd)\":/workspace -w /workspace \
        mimo-audio:latest python finetune_encoder_v2.py \
        --data-dir \"$DATA_DIR\" \
        --tokenizer-path \"$TOKENIZER\" \
        --output-dir \"$OUTPUT_DIR\" \
        --rank $RANK \
        --alpha $ALPHA \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation $GRAD_ACCUM \
        --epochs $EPOCHS \
        --lr $LR \
        --save-every 10
    
    EXIT_CODE=\$?
    
    echo ''
    echo '=========================================='
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '✅ Training Completed Successfully!'
        echo '=========================================='
        echo ''
        echo '📊 查看結果:'
        echo \"   cat $OUTPUT_DIR/training_history.json\"
        echo ''
        echo '🧪 測試模型:'
        echo \"   ./test_inference_docker.sh --no-tmux\"
    else
        echo '❌ Training Failed (Exit Code: '\$EXIT_CODE')'
        echo '=========================================='
    fi
    
    echo ''
    echo '按 Enter 繼續...'
    read
"

echo "✅ Training started in tmux session '$SESSION_NAME'"
echo ""
echo "📊 監控訓練:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "📈 查看輸出目錄:"
echo "   watch -n 5 'ls -lh $OUTPUT_DIR/'"
echo ""
echo "🔍 實時查看 log (如果有):"
echo "   tail -f $OUTPUT_DIR/training.log"
echo ""
