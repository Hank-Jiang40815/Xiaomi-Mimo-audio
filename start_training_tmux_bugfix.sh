#!/bin/bash
# 在 tmux 中啟動 Bug 修復驗證訓練

SESSION_NAME="finetune_bugfix_test"

# 檢查 session 是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "❌ Tmux session '$SESSION_NAME' 已存在"
    echo "請使用以下命令查看："
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "或刪除舊 session："
    echo "  tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "🚀 在 tmux session '$SESSION_NAME' 中啟動訓練..."
echo ""

# 創建新的 tmux session 並在其中執行訓練
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd /home/sbplab/Hank/MiMo-Audio" C-m
tmux send-keys -t $SESSION_NAME "./start_training_r32_e20_bugfix.sh" C-m

echo "✅ 訓練已在 tmux session '$SESSION_NAME' 中啟動"
echo ""
echo "📊 查看訓練進度："
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "⚠️  離開 tmux（不中斷訓練）："
echo "  按 Ctrl+B 然後按 D"
echo ""
echo "🔍 檢查訓練狀態："
echo "  tmux ls"
echo ""
