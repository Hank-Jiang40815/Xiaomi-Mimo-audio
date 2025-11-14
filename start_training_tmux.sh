#!/bin/bash
# Fine-tuning 訓練腳本 - 使用 tmux 保護
# 可防止：電源中斷、網路斷線、SSH 斷線

SESSION_NAME="finetune_r32_e100"

echo "=========================================="
echo "🚀 準備在 tmux 中啟動訓練"
echo "Session 名稱: $SESSION_NAME"
echo "=========================================="
echo ""

# 檢查是否已有同名 session
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  發現已存在的 tmux session: $SESSION_NAME"
    echo ""
    echo "選項："
    echo "  1. 重新連接到現有 session: tmux attach -t $SESSION_NAME"
    echo "  2. 刪除舊 session 並重新開始: tmux kill-session -t $SESSION_NAME"
    echo ""
    read -p "是否要刪除舊 session 並重新開始？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  刪除舊 session..."
        tmux kill-session -t $SESSION_NAME
    else
        echo "❌ 取消啟動"
        exit 1
    fi
fi

echo "📝 創建 tmux session..."
echo ""

# 創建新的 tmux session 並在其中執行訓練
tmux new-session -d -s $SESSION_NAME

# 在 tmux session 中執行訓練命令
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo '🎯 Fine-tuning 訓練開始'" C-m
tmux send-keys -t $SESSION_NAME "echo '配置：Rank-32, 100 Epochs, Batch-8'" C-m
tmux send-keys -t $SESSION_NAME "echo '預計訓練時間：70-90 分鐘'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo '🚀 開始訓練...'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# 執行實際的訓練命令
tmux send-keys -t $SESSION_NAME "docker run --gpus all --rm -v \"\$(pwd)\":/workspace -w /workspace mimo-audio:latest python finetune_encoder.py --train-split data/splits/finetune_optical/train.json --val-split data/splits/finetune_optical/val.json --batch-size 8 --gradient-accumulation-steps 8 --epochs 100 --lora-rank 32 --lora-alpha 64 --lr 5e-5 --output-dir ./outputs/optical_lora_r32_e100 --num-workers 4 2>&1 | tee ./outputs/optical_lora_r32_e100/training.log" C-m

echo ""
echo "✅ Tmux session 創建成功！"
echo ""
echo "=========================================="
echo "📋 重要指令："
echo "=========================================="
echo ""
echo "🔗 連接到訓練 session:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "⌨️  在 tmux 中的快捷鍵:"
echo "   Ctrl+B 然後 D     - 離開 session (訓練繼續執行)"
echo "   Ctrl+B 然後 [     - 進入捲動模式 (可以往上看 log)"
echo "   Ctrl+B 然後 ?     - 顯示所有快捷鍵"
echo ""
echo "📊 查看訓練狀態 (不進入 session):"
echo "   tmux capture-pane -t $SESSION_NAME -p | tail -20"
echo ""
echo "🔍 監控 log 檔案:"
echo "   tail -f ./outputs/optical_lora_r32_e100/training.log"
echo ""
echo "🗑️  停止訓練並刪除 session:"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "📋 列出所有 tmux sessions:"
echo "   tmux ls"
echo ""
echo "=========================================="
echo ""
echo "💡 提示："
echo "   即使你關閉 SSH 連線、電源中斷或網路斷線，"
echo "   訓練仍會在背景繼續執行！"
echo ""
echo "🎯 現在連接到 session 查看訓練進度:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
