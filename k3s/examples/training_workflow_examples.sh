#!/bin/bash

# ============================================
# K3S GPU Pod 連續訓練工作流示例
# ============================================
# 場景：需要連續訓練多個模型或實驗

# 【步驟 1】首先進入 Pod
echo "=== 連接到 GPU Pod ==="
kubectl exec -it gpu-pod-working -- bash

# 【進入 Pod 後執行以下命令】

# ============================================
# 方案 A：順序訓練（一個接一個）
# ============================================

echo "=== 方案 A：順序訓練 ==="

# 建立訓練目錄
mkdir -p /root/training_logs
cd /root/training_logs

# 訓練模型 1
echo "開始訓練 Model 1..."
python train.py --model model_1 --epochs 100 --output ./model_1_checkpoint \
  | tee model_1_training.log

# 檢查訓練是否成功
if [ $? -eq 0 ]; then
  echo "✅ Model 1 訓練完成"
else
  echo "❌ Model 1 訓練失敗"
  exit 1
fi

# 訓練模型 2
echo "開始訓練 Model 2..."
python train.py --model model_2 --epochs 100 --output ./model_2_checkpoint \
  | tee model_2_training.log

# 訓練模型 3
echo "開始訓練 Model 3..."
python train.py --model model_3 --epochs 100 --output ./model_3_checkpoint \
  | tee model_3_training.log

echo "✅ 所有模型訓練完成！"

# ============================================
# 方案 B：批量訓練（使用迴圈）
# ============================================

echo "=== 方案 B：批量訓練 ==="

# 定義訓練列表
MODELS=("model_1" "model_2" "model_3" "model_4" "model_5")
EPOCHS=(50 100 150 50 100)

# 迴圈訓練
for i in "${!MODELS[@]}"; do
  model=${MODELS[$i]}
  epoch=${EPOCHS[$i]}
  
  echo "🔄 訓練 $model (Epochs: $epoch)"
  
  python train.py \
    --model "$model" \
    --epochs "$epoch" \
    --output "./checkpoints/$model" \
    --batch_size 32 \
    --learning_rate 0.001 \
    2>&1 | tee "./logs/${model}_training.log"
  
  if [ $? -eq 0 ]; then
    echo "✅ $model 訓練完成"
    # 可選：評估模型
    python evaluate.py --checkpoint "./checkpoints/$model" | tee "./logs/${model}_eval.log"
  else
    echo "❌ $model 訓練失敗，停止"
    exit 1
  fi
  
  # 等待一下，讓 GPU 冷卻（可選）
  echo "等待 10 秒..."
  sleep 10
done

echo "✅ 所有訓練和評估完成！"

# ============================================
# 方案 C：並行訓練（需要多個 Pod）
# ============================================

echo "=== 方案 C：並行訓練（推薦用於多 GPU） ==="

# 這需要在主機端執行，不是在 Pod 內

# 建立多個 Pod 配置
cat > /tmp/pod_model1.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod-model-1
spec:
  restartPolicy: OnFailure
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command:
      - /bin/bash
      - -c
      - |
        cd /root/training
        python train.py --model model_1 --epochs 100 | tee model_1.log
        while true; do sleep 3600; done
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
EOF

# 啟動多個 Pod
kubectl apply -f /tmp/pod_model1.yaml
# ... 類似地建立更多 Pod

# ============================================
# 方案 D：使用 Kubernetes Job（最佳實踐）
# ============================================

echo "=== 方案 D：Kubernetes Job（推薦） ==="

cat > /tmp/training_job.yaml << 'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-training-job
spec:
  completions: 5  # 訓練 5 個模型
  parallelism: 1  # 一次訓練 1 個（改為 5 就能並行）
  activeDeadlineSeconds: 86400  # 24 小時期限
  backoffLimit: 3  # 失敗 3 次後停止
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: nvidia/cuda:12.2.0-base-ubuntu22.04
        command:
          - /bin/bash
          - -c
          - |
            python train.py --job-index $((RANDOM % 5 + 1)) --epochs 100
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
EOF

kubectl apply -f /tmp/training_job.yaml

# ============================================
# 實用輔助指令
# ============================================

echo "=== 實用輔助指令 ==="

# 1. 在後台執行（即使 SSH 斷開也繼續）
# 使用 nohup 或 screen/tmux

echo "方案 1：使用 nohup（在 Pod 內）"
cat > run_training.sh << 'EOF'
#!/bin/bash
nohup python train.py --model model_1 --epochs 100 > training.log 2>&1 &
echo "訓練已在後台啟動，PID: $!"
EOF

echo "方案 2：使用 tmux（在 Pod 內建立會話）"
cat > run_with_tmux.sh << 'EOF'
#!/bin/bash
tmux new-session -d -s training
tmux send-keys -t training "cd /root/training && python train.py --epochs 100" Enter
echo "訓練在 tmux 會話 'training' 中運行"
tmux attach -t training
EOF

# 2. 監控訓練進度
echo "在另一個終端監控："
echo "kubectl logs -f gpu-pod-working"

# 3. 定期檢查 GPU 使用情況
echo "watch -n 5 'kubectl exec gpu-pod-working -- nvidia-smi'"

# 4. 在 Pod 內檢查進程
echo "ps aux | grep python"

# 5. 檢查磁盤空間（避免滿容量）
echo "df -h"

# ============================================
# 進階：監控和自動化
# ============================================

echo "=== 監控訓練進度 ==="

# 建立監控腳本（在主機上執行）
cat > monitor_training.sh << 'MONITOR'
#!/bin/bash

while true; do
  clear
  echo "=== 訓練監控 $(date) ==="
  echo ""
  
  # Pod 狀態
  echo "【Pod 狀態】"
  kubectl get pod gpu-pod-working
  echo ""
  
  # GPU 使用情況
  echo "【GPU 使用】"
  kubectl exec gpu-pod-working -- nvidia-smi
  echo ""
  
  # 訓練日誌（最後 5 行）
  echo "【訓練日誌】"
  kubectl logs gpu-pod-working --tail=5
  echo ""
  
  # 運行進程
  echo "【運行進程】"
  kubectl exec gpu-pod-working -- ps aux | grep python | grep -v grep
  echo ""
  
  echo "5 秒後更新..."
  sleep 5
done
MONITOR

chmod +x monitor_training.sh

# ============================================
# 檢查清單：開始連續訓練前
# ============================================

echo "=== 連續訓練前檢查清單 ==="
cat << 'CHECKLIST'
☐ 1. 確認 GPU Pod 已啟動
     kubectl get pods
     
☐ 2. 確認 GPU 可用
     kubectl exec gpu-pod-working -- nvidia-smi
     
☐ 3. 將訓練資料複製到 Pod
     kubectl cp ./data gpu-pod-working:/root/data
     kubectl cp ./scripts gpu-pod-working:/root/scripts
     
☐ 4. 創建訓練日誌目錄
     kubectl exec gpu-pod-working -- mkdir -p /root/training/logs /root/training/checkpoints
     
☐ 5. 如果訓練超過 3 小時，延長期限
     kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'
     
☐ 6. 準備訓練腳本和配置檔案
     
☐ 7. 測試單個訓練流程（確保沒有錯誤）
     
☐ 8. 啟動監控終端
     bash monitor_training.sh
     
☐ 9. 啟動訓練
     
☐ 10. 定期檢查進度和 GPU 狀態
CHECKLIST

# ============================================
# 故障排除
# ============================================

echo "=== 常見問題和解決方案 ==="

cat << 'TROUBLESHOOTING'

【問題】訓練中途 Pod 自動停止
【原因】3 小時期限已到
【解決】
  kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'

【問題】GPU 內存不足
【原因】batch_size 太大或多個訓練同時運行
【解決】
  - 減小 batch_size
  - 確保只有一個訓練在運行
  - kubectl exec gpu-pod-working -- nvidia-smi

【問題】訓練進度很慢
【原因】數據 I/O 瓶頸或其他進程佔用 GPU
【解決】
  - 檢查 GPU 使用率：kubectl exec gpu-pod-working -- nvidia-smi
  - 檢查磁盤 I/O：kubectl exec gpu-pod-working -- iostat -x
  - 預加載數據到 GPU 內存

【問題】訓練日誌太多，占滿磁盤
【原因】沒有定期清理日誌
【解決】
  kubectl exec gpu-pod-working -- bash -c 'find /root/training/logs -mtime +7 -delete'

【問題】Pod 意外終止
【原因】Out of Memory 或容器崩潰
【解決】
  kubectl describe pod gpu-pod-working
  kubectl logs gpu-pod-working --previous

TROUBLESHOOTING
