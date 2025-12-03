#!/bin/bash

# ============================================
# 快速啟動 Pod 訓練（簡化版本）
# ============================================
# 最快的啟動方式

POD_NAME="gpu-pod-working"
TMUX_SESSION="pod_training"

echo "🚀 快速啟動 Pod 訓練..."

# 1. 複製訓練腳本
echo "📁 複製文件..."
kubectl cp ./train_batch.sh "$POD_NAME":/root/ 2>/dev/null && echo "✅ 訓練腳本已複製" || echo "⚠️ 腳本可能已存在"

# 2. 在 Pod 內建立 tmux 會話
echo "🔧 設置 tmux..."
kubectl exec "$POD_NAME" -- tmux new-session -d -s "$TMUX_SESSION" -c /root 2>/dev/null || echo "⚠️ 會話已存在，使用現有的"

# 3. 在 tmux 內啟動訓練
echo "🎯 啟動訓練..."
kubectl exec "$POD_NAME" -- tmux send-keys -t "$TMUX_SESSION" "bash /root/train_batch.sh" Enter

# 4. 連接到會話
echo "📊 連接到監控..."
echo ""
echo "════════════════════════════════════════"
echo "訓練已啟動！"
echo "════════════════════════════════════════"
echo ""

kubectl exec -it "$POD_NAME" -- tmux attach-session -t "$TMUX_SESSION"
