#!/bin/bash
# Test finetuned encoder inference in Docker (with optional tmux)
# 在 Docker 容器中測試微調後的 Encoder
# 
# 使用方式:
#   ./test_inference_docker.sh              # 使用 tmux (背景執行)
#   ./test_inference_docker.sh --no-tmux    # 直接執行 (前景)

USE_TMUX=true
if [ "$1" == "--no-tmux" ]; then
    USE_TMUX=false
fi

SESSION_NAME="test_inference"

echo "=========================================="
echo "測試微調後的 Encoder Inference"
echo "=========================================="
echo ""

# 配置 (可透過環境變數覆蓋)
CHECKPOINT="${CHECKPOINT:-outputs/optical_lora_r32_e100/best_model.pt}"
TOKENIZER="${TOKENIZER:-models/MiMo-Audio-Tokenizer}"
INPUT="${INPUT:-examples/optical/mix/boy1_WOLDV_050.wav}"
OUTPUT="${OUTPUT:-outputs/test_inference/enhanced_050.wav}"

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
mkdir -p "$(dirname "$OUTPUT")"

# Docker 執行指令
DOCKER_CMD="docker run --gpus all --rm \
    -v \"\$(pwd)\":/workspace -w /workspace \
    mimo-audio:latest python test_finetuned_inference.py \
    --checkpoint \"$CHECKPOINT\" \
    --tokenizer-path \"$TOKENIZER\" \
    --input \"$INPUT\" \
    --output \"$OUTPUT\" \
    --device cuda"

if [ "$USE_TMUX" = true ]; then
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
        
        $DOCKER_CMD
        
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
    echo "   執行 'tmux attach -t $SESSION_NAME' 查看進度"
else
    # 直接執行模式（前景）
    echo "🚀 在 Docker 中直接執行測試..."
    echo ""
    
    eval "$DOCKER_CMD"
    
    EXIT_CODE=$?
    
    echo ""
    echo "=========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ 測試完成！"
        echo "=========================================="
        echo ""
        
        if [ -f "$OUTPUT" ]; then
            FILE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
            echo "📁 輸出檔案: $OUTPUT"
            echo "📊 大小: $FILE_SIZE"
            echo ""
            echo "🎧 播放指令:"
            echo "   原始: ffplay -autoexit -nodisp $INPUT"
            echo "   增強: ffplay -autoexit -nodisp $OUTPUT"
        else
            echo "❌ 輸出檔案未生成"
            exit 1
        fi
    else
        echo "❌ 測試失敗 (Exit Code: $EXIT_CODE)"
        echo "=========================================="
        exit $EXIT_CODE
    fi
fi

echo "✅ tmux session '$SESSION_NAME' 已啟動"
echo ""
echo "📺 連接到 session 查看輸出："
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "或等待幾秒後自動連接..."
sleep 2
tmux attach -t "$SESSION_NAME"
