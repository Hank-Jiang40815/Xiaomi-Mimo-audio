#!/bin/bash

# ============================================
# K3S 訓練 - 快速參考卡片
# ============================================

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║                  K3S GPU Pod 訓練 - 快速參考卡片                       ║
╚════════════════════════════════════════════════════════════════════════╝

📂 目錄位置
  k3s/
  ├── configs/      Pod 配置
  ├── scripts/      訓練腳本
  ├── examples/     詳細範例
  ├── docs/         文檔指南
  └── README.md     完整說明

🚀 最快啟動（3 步）
  1. cd k3s/scripts
  2. bash quick_start_training.sh
  3. 進入 tmux 監控（自動進入）

⚙️ 修改訓練配置
  vim k3s/scripts/train_batch.sh
  修改 CONFIGS 和 CONFIG_PARAMS

📊 查看進度
  方法 1：進入 tmux（已自動進入）
  方法 2：新終端監控
    bash k3s/scripts/monitor_and_recover.sh monitor
  方法 3：查看日誌
    kubectl logs -f gpu-pod-working

🔑 重要命令
  
  進入 Pod
    kubectl exec -it gpu-pod-working -- bash
  
  查看 GPU
    kubectl exec gpu-pod-working -- nvidia-smi
  
  連接 tmux
    kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training
  
  停止訓練
    kubectl exec gpu-pod-working -- pkill -f python
  
  備份結果
    bash k3s/scripts/monitor_and_recover.sh backup
  
  查看狀態
    bash launch_pod_training.sh --status

⏱️ 重要提醒
  
  ⚠️ Pod 默認 3 小時後自動停止
    延長期限：kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'
  
  💾 SSH 斷線不影響訓練
    Pod 和 tmux 繼續運行
  
  🔄 隨時可以重新連接
    kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training

📖 查看完整指南
  bash k3s/docs/POD_TRAINING_GUIDE.sh

════════════════════════════════════════════════════════════════════════

EOF
