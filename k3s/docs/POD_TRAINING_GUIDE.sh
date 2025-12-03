#!/bin/bash

# ============================================
# Pod + tmux 訓練常用指令速查表
# ============================================

echo "
╔════════════════════════════════════════════════════════════════════════╗
║         K3S GPU Pod + tmux 訓練一站式完整指南                          ║
╚════════════════════════════════════════════════════════════════════════╝

【最快啟動】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ 三步快速啟動（推薦）

  bash quick_start_training.sh

  說明：
  - 自動複製訓練腳本到 Pod
  - 自動建立 tmux 會話
  - 自動啟動訓練並進入監控

2️⃣ 互動式菜單（功能最全）

  bash launch_pod_training.sh

  可選項：
  --background  後台啟動（不進入監控）
  --monitor     連接到監控
  --logs        查看日誌
  --status      檢查狀態
  --stop        停止訓練

【手動操作步驟】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步驟 1：複製訓練腳本到 Pod
  kubectl cp ./train_batch.sh gpu-pod-working:/root/

步驟 2：進入 Pod
  kubectl exec -it gpu-pod-working -- bash

步驟 3：在 Pod 內建立 tmux 會話（可選但推薦）
  tmux new-session -s training

步驟 4：在 tmux 內執行訓練
  bash /root/train_batch.sh

步驟 5：分離 tmux（訓練繼續運行）
  按 Ctrl+B，然後按 D

步驟 6：重新連接（查看進度）
  tmux attach-session -s training

【常用指令速查】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 查看 Pod 狀態
  kubectl get pods
  kubectl describe pod gpu-pod-working
  kubectl logs -f gpu-pod-working

🖥️ 進入 Pod
  kubectl exec -it gpu-pod-working -- bash

📁 文件傳輸
  # 主機 → Pod
  kubectl cp /local/path gpu-pod-working:/pod/path
  
  # Pod → 主機
  kubectl cp gpu-pod-working:/pod/path /local/path

⏱️ 查看 GPU 使用
  kubectl exec gpu-pod-working -- nvidia-smi
  kubectl exec gpu-pod-working -- watch nvidia-smi

🔄 tmux 常用快捷鍵（在 Pod 內）
  Ctrl+B, D    分離會話（訓練繼續）
  Ctrl+B, L    切換上一個會話
  Ctrl+B, [    進入複製模式
  Ctrl+B, :    進入命令模式
  exit         退出當前窗格

📝 監控訓練
  # 方案 1：查看日誌
  kubectl logs -f gpu-pod-working
  
  # 方案 2：進入 Pod 查看進程
  kubectl exec gpu-pod-working -- ps aux | grep python
  
  # 方案 3：監控腳本（從主機）
  bash monitor_and_recover.sh monitor

【故障排除】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ 訓練中斷了怎麼辦？

  # 檢查 Pod 狀態
  kubectl describe pod gpu-pod-working

  # 檢查日誌
  kubectl logs gpu-pod-working --previous

  # 檢查事件
  kubectl get events --sort-by='.lastTimestamp' | tail -10

  # 重啟 Pod
  kubectl delete pod gpu-pod-working
  kubectl apply -f gpu-pod-working.yaml

❓ 訓練超過 3 小時期限？

  # 檢查剩餘時間
  kubectl get pod gpu-pod-working -o jsonpath='{.spec.activeDeadlineSeconds}'

  # 延長期限（改為無限期）
  kubectl patch pod gpu-pod-working -p '{\"spec\":{\"activeDeadlineSeconds\":null}}'

  # 或者改為 24 小時（86400 秒）
  kubectl patch pod gpu-pod-working -p '{\"spec\":{\"activeDeadlineSeconds\":86400}}'

❓ tmux 會話找不到？

  # 查看現有會話
  kubectl exec gpu-pod-working -- tmux list-sessions

  # 創建新會話
  kubectl exec gpu-pod-working -- tmux new-session -d -s training

  # 進入會話
  kubectl exec -it gpu-pod-working -- tmux attach-session -s training

❓ GPU 內存不足？

  # 檢查 GPU 使用
  kubectl exec gpu-pod-working -- nvidia-smi

  # 檢查運行進程
  kubectl exec gpu-pod-working -- ps aux | grep python

  # 停止訓練
  kubectl exec gpu-pod-working -- pkill -f python

❓ 無法複製文件到 Pod？

  # 檢查 Pod 是否運行
  kubectl get pod gpu-pod-working

  # 檢查目標目錄權限
  kubectl exec gpu-pod-working -- ls -la /root/

  # 建立目錄
  kubectl exec gpu-pod-working -- mkdir -p /root/data

【工作流範例】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【場景 1】我想快速啟動訓練並監控

  # 在主機上執行：
  bash quick_start_training.sh

【場景 2】我想後台運行，訓練完後再檢查

  # 在主機上執行：
  bash launch_pod_training.sh --background

  # 訓練完成後檢查結果：
  kubectl exec gpu-pod-working -- ls -la /root/training_results/

【場景 3】我想長期運行多個訓練

  # 準備訓練配置文件（修改 train_batch.sh）
  # 然後執行：
  bash quick_start_training.sh

  # 在另一個終端監控進度：
  watch -n 5 'kubectl exec gpu-pod-working -- nvidia-smi'

【場景 4】訓練中途想中斷或修改

  # 進入 Pod 的 tmux 會話：
  bash launch_pod_training.sh --monitor

  # 或者直接進入：
  kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training

  # 在 tmux 內按 Ctrl+C 停止訓練
  # 或按 Ctrl+B, D 分離會話

【場景 5】我想比較不同的訓練結果

  # 備份結果：
  bash monitor_and_recover.sh backup

  # 結果位置：
  /tmp/training_backup_<timestamp>/

【進階技巧】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 同時監控多個訓練

  # 終端 1：進入 tmux 看實時輸出
  kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training

  # 終端 2：監控 GPU 使用
  watch -n 2 'kubectl exec gpu-pod-working -- nvidia-smi'

  # 終端 3：查看日誌
  kubectl logs -f gpu-pod-working

💡 自動定期備份訓練結果

  # 執行自動備份（每 5 分鐘）
  bash monitor_and_recover.sh auto-save 300

💡 分析訓練日誌

  # 查看訓練速度
  kubectl exec gpu-pod-working -- tail -100 /root/training_results/logs/*.log

  # 搜索特定配置的結果
  kubectl exec gpu-pod-working -- grep -r \"success\" /root/training_results/logs/

【必讀：tmux vs Pod 的區別】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tmux（在 Pod 內運行）
  ✅ 即使 SSH 斷開，訓練也繼續
  ✅ 可以分離和重新連接會話
  ✅ 支持多窗格和多窗口
  ✅ 可以複製歷史輸出

Pod（容器級別）
  ✅ 自動資源隔離
  ✅ 自動 GPU 分配
  ✅ 失敗自動重啟
  ✅ 支持多個並行 Pod

最佳實踐：在 Pod 內使用 tmux = 兩全其美！

【檢查清單：開始訓練前】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

☐ Pod 已啟動
  kubectl get pod gpu-pod-working

☐ GPU 可用
  kubectl exec gpu-pod-working -- nvidia-smi

☐ 訓練腳本已準備
  ls -la train_batch.sh

☐ 訓練配置已編輯
  vim train_batch.sh

☐ 有足夠的磁盤空間
  kubectl exec gpu-pod-working -- df -h

☐ 備份了舊的訓練結果（如需要）
  bash monitor_and_recover.sh backup

☐ 如果訓練超過 3 小時，延長期限
  kubectl patch pod gpu-pod-working -p '{\"spec\":{\"activeDeadlineSeconds\":null}}'

✅ 準備完成，現在執行：
  bash quick_start_training.sh

═══════════════════════════════════════════════════════════════════════════

需要幫助？查看詳細指南：
  cat K3S_GPU_POD_COMMANDS.md
  cat training_workflow_examples.sh
"
