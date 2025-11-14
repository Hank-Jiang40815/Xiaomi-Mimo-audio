#!/bin/bash
# 快速查看訓練狀態腳本

SESSION_NAME="finetune_r32_e100"
LOG_FILE="./outputs/optical_lora_r32_e100/training.log"

echo "=========================================="
echo "📊 Fine-tuning 訓練狀態"
echo "=========================================="
echo ""

# 檢查 tmux session 是否存在
if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "❌ Tmux session '$SESSION_NAME' 不存在"
    echo ""
    echo "可能的原因："
    echo "  1. 訓練尚未啟動"
    echo "  2. 訓練已經完成"
    echo "  3. Session 被意外關閉"
    echo ""
    echo "請執行: ./start_training_tmux.sh"
    exit 1
fi

echo "✅ Tmux session 運行中"
echo ""

# 顯示 GPU 狀態
echo "🖥️  GPU 狀態:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# 顯示最近的訓練 log
if [ -f "$LOG_FILE" ]; then
    echo "📝 最近的訓練記錄 (最後 30 行):"
    echo "=========================================="
    tail -30 "$LOG_FILE"
    echo "=========================================="
    echo ""
    
    # 統計已完成的 epochs
    COMPLETED_EPOCHS=$(grep -c "Train Loss:" "$LOG_FILE")
    echo "📈 已完成 Epochs: $COMPLETED_EPOCHS / 100"
    
    # 顯示最新的 loss
    echo ""
    echo "📊 最近 5 個 Epoch 的 Loss:"
    grep -E "Train Loss:|Val Loss:" "$LOG_FILE" | tail -10
else
    echo "⚠️  Log 檔案尚未生成"
fi

echo ""
echo "=========================================="
echo "💡 更多操作："
echo "   連接到 session:  tmux attach -t $SESSION_NAME"
echo "   實時查看 log:    tail -f $LOG_FILE"
echo "   停止訓練:        tmux kill-session -t $SESSION_NAME"
echo "=========================================="
