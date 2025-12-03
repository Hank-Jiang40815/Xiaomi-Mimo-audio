# K3S GPU Pod 快速指令手冊

## 📋 查看 Pod 狀態

```bash
# 查看所有 Pod
kubectl get pods

# 詳細查看 gpu-pod-working
kubectl describe pod gpu-pod-working

# 即時監控 Pod 狀態
kubectl get pods -w

# 查看 Pod 的 GPU 使用情況
kubectl exec gpu-pod-working -- nvidia-smi
```

## 🚀 進入 Pod 操作

```bash
# 進入 Pod 的 bash 終端
kubectl exec -it gpu-pod-working -- bash

# 執行單次命令（不進入終端）
kubectl exec gpu-pod-working -- python your_script.py

# 執行含 GPU 的 Python 腳本
kubectl exec gpu-pod-working -- python -c "import torch; print(torch.cuda.is_available())"
```

## 📁 檔案傳輸

```bash
# 從主機複製檔案到 Pod
kubectl cp /path/to/local/file gpu-pod-working:/path/in/pod

# 從 Pod 複製檔案到主機
kubectl cp gpu-pod-working:/path/in/pod /path/to/local/file

# 複製整個目錄
kubectl cp /path/to/local/dir gpu-pod-working:/path/in/pod -r
```

## 📊 查看日誌

```bash
# 查看 Pod 的完整日誌
kubectl logs gpu-pod-working

# 即時跟蹤日誌（類似 tail -f）
kubectl logs -f gpu-pod-working

# 查看最後 100 行日誌
kubectl logs gpu-pod-working --tail=100
```

## 🔧 Pod 生命週期管理

```bash
# 刪除 Pod（會自動重啟，因為有 OnFailure 策略）
kubectl delete pod gpu-pod-working

# 重新建立 Pod
kubectl apply -f /home/sbplab/Hank/MiMo-Audio/gpu-pod-working.yaml

# 強制刪除 Pod（立即終止）
kubectl delete pod gpu-pod-working --grace-period=0 --force
```

## ⚠️ 修改 Pod 限制

```bash
# 移除 3 小時期限（永久運行）
kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'

# 設定新的期限（例如 24 小時 = 86400 秒）
kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":86400}}'

# 查看目前的期限設定
kubectl get pod gpu-pod-working -o jsonpath='{.spec.activeDeadlineSeconds}'
```

## 📦 資源監控

```bash
# 查看 Pod 的資源使用
kubectl top pod gpu-pod-working

# 查看所有 Pod 的資源使用
kubectl top pods

# 查看 GPU 分配狀況
kubectl describe node sbplab | grep -A 10 "nvidia.com/gpu"
```

## 🔄 常見工作流

### 場景 1：運行長時間的訓練
```bash
# 1. 進入 Pod
kubectl exec -it gpu-pod-working -- bash

# 2. 在 Pod 內運行訓練（會持續 3 小時）
python train.py

# 3. 或者用後台運行（即使斷開連線也繼續）
nohup python train.py > training.log 2>&1 &
```

### 場景 2：運行 Jupyter Notebook
```bash
# 1. 進入 Pod
kubectl exec -it gpu-pod-working -- bash

# 2. 在 Pod 內啟動 Jupyter
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

# 3. 複製 token 到瀏覽器打開筆記本
```

### 場景 3：監控執行中的任務
```bash
# 終端 1：查看實時日誌
kubectl logs -f gpu-pod-working

# 終端 2：監控 GPU 使用
watch -n 1 'kubectl exec gpu-pod-working -- nvidia-smi'

# 終端 3：進入 Pod 互動
kubectl exec -it gpu-pod-working -- bash
```

## ⏰ 重要提醒

| 項目 | 當前設定 |
|-----|--------|
| Pod 啟動時間 | 2025-12-03 14:23:09 |
| 自動終止時間 | 2025-12-03 17:23:09（3 小時後） |
| GPU 數量 | 1 顆 RTX 5090 |
| 重啟策略 | 失敗時自動重啟 |

**如果 3 小時內工作還沒完成，執行**：
```bash
kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'
```

## 🆘 遇到問題

```bash
# Pod 狀態異常？查看詳細事件
kubectl describe pod gpu-pod-working

# 容器崩潰？查看容器日誌
kubectl logs gpu-pod-working

# GPU 識別不到？進入 Pod 檢查
kubectl exec gpu-pod-working -- nvidia-smi

# 需要重啟 Pod
kubectl delete pod gpu-pod-working
kubectl apply -f /home/sbplab/Hank/MiMo-Audio/gpu-pod-working.yaml
```
