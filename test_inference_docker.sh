#!/bin/bash
# Test finetuned encoder inference in Docker + tmux
# 在 Docker 容器中測試微調後的 Encoder

SESSION_NAME="test_inference"

echo "=========================================="
echo "測試微調後的 Encoder Inference"
echo "=========================================="
echo ""

# 配置
CHECKPOINT="outputs/optical_lora_r32_e100/best_model.pt"
TOKENIZER="models/MiMo-Audio-Tokenizer"
INPUT="examples/optical/mix/boy1_WOLDV_050.wav"
OUTPUT="outputs/test_inference/enhanced_050.wav"

# 檢查文件
echo "📋 檢查必要文件..."

if [ ! -f "$CHECKPOINT" ]; then
    echo "⚠️  best_model.pt 不存在，嘗試使用 checkpoint_epoch_2.pt"
    CHECKPOINT="outputs/optical_lora_r32_e100/checkpoint_epoch_2.pt"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 找不到任何可用的 checkpoint"
    echo "   請確認訓練已完成並產生了 checkpoint"
    exit 1
fi

if [ ! -d "$TOKENIZER" ]; then
    echo "❌ Tokenizer 不存在: $TOKENIZER"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "❌ 測試音訊不存在: $INPUT"
    exit 1
fi

echo "✅ Checkpoint: $CHECKPOINT"
echo "✅ Tokenizer: $TOKENIZER"
echo "✅ Input: $INPUT"
echo ""

# 建立輸出目錄
mkdir -p outputs/test_inference

# 檢查是否已有同名 tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION_NAME' 已存在"
    echo "   選項："
    echo "   1. tmux attach -t $SESSION_NAME  # 連接到現有 session"
    echo "   2. tmux kill-session -t $SESSION_NAME  # 刪除後重新執行此腳本"
    exit 1
fi

echo "🚀 在 Docker + tmux 中啟動測試..."
echo ""
echo "指令："
echo "  連接: tmux attach -t $SESSION_NAME"
echo "  分離: Ctrl+B 然後按 D"
echo ""

# 在 tmux 中執行 docker
tmux new-session -d -s "$SESSION_NAME" bash -c "
    echo '=========================================='
    echo '🐳 Docker 容器中執行 Inference 測試'
    echo '=========================================='
    echo ''
    echo '📦 Checkpoint: $CHECKPOINT'
    echo '🎵 Input: $INPUT'
    echo '📤 Output: $OUTPUT'
    echo ''
    echo '開始測試...'
    echo ''
    
    docker run --gpus all --rm \
        -v \"\$(pwd)\":/workspace -w /workspace \
        mimo-audio:latest python test_finetuned_inference.py \
        --checkpoint \"$CHECKPOINT\" \
        --tokenizer-path \"$TOKENIZER\" \
        --input \"$INPUT\" \
        --output \"$OUTPUT\" \
        --device cuda
    
    EXIT_CODE=\$?
    
    echo ''
    echo '=========================================='
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '✅ 測試完成！'
        echo '=========================================='
        echo ''
        echo '📊 結果檔案:'
        echo \"   $OUTPUT\"
        echo ''
        echo '🎧 播放指令:'
        echo \"   原始: ffplay -autoexit -nodisp $INPUT\"
        echo \"   增強: ffplay -autoexit -nodisp $OUTPUT\"
        echo ''
        echo '📈 檔案大小比較:'
        ls -lh \"$INPUT\" \"$OUTPUT\" 2>/dev/null || echo '   (輸出檔案不存在)'
    else
        echo '❌ 測試失敗 (Exit Code: '\$EXIT_CODE')'
        echo '=========================================='
    fi
    
    echo ''
    echo '按 Enter 繼續...'
    read
"

echo "✅ tmux session '$SESSION_NAME' 已啟動"
echo ""
echo "📺 連接到 session 查看輸出："
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "或等待幾秒後自動連接..."
sleep 2
tmux attach -t "$SESSION_NAME"
