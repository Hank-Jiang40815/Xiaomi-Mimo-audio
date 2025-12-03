# K3S GPU Pod 訓練管理系統

## 📁 目錄結構

```
k3s/
├── configs/              # Kubernetes Pod 配置
│   └── gpu-pod-working.yaml
├── scripts/              # 訓練管理腳本
│   ├── train_batch.sh            # 批量訓練腳本
│   ├── quick_start_training.sh   # 快速啟動（推薦）
│   ├── launch_pod_training.sh    # 完整功能菜單
│   └── monitor_and_recover.sh    # 監控和恢復
├── examples/             # 詳細範例和說明
│   └── training_workflow_examples.sh
├── docs/                 # 文檔和指南
│   ├── K3S_GPU_POD_COMMANDS.md      # 命令參考
│   └── POD_TRAINING_GUIDE.sh        # 完整指南
└── README.md             # 本文件
```

## 🚀 快速開始（3 選 1）

### 方案 1：最簡單（推薦 ⭐⭐⭐）
```bash
cd k3s/scripts
bash quick_start_training.sh
```
- ✅ 自動複製、建立 tmux、啟動訓練、進入監控
- ⏱️ 最快上手

### 方案 2：完整功能（推薦 ⭐⭐）
```bash
cd k3s/scripts
bash launch_pod_training.sh
```
- ✅ 交互式菜單、後台運行、監控、日誌查看
- 🎮 功能最全

### 方案 3：查看指南
```bash
bash k3s/docs/POD_TRAINING_GUIDE.sh
```
- 📖 查看所有命令和範例

## 📋 前置準備

### 1. 確認 Pod 已啟動
```bash
kubectl get pods
# 應該看到 gpu-pod-working 運行中
```

### 2. 如果 Pod 不存在，建立它
```bash
kubectl apply -f k3s/configs/gpu-pod-working.yaml
kubectl get pods -w  # 等待 Pod Running
```

### 3. 編輯訓練配置
```bash
vim k3s/scripts/train_batch.sh

# 修改這部分：
# declare -a CONFIGS=(
#   "exp_a_r32_e20"
#   "exp_a_r32_e50"
#   ...
# )
```

### 4. 延長 Pod 期限（如果訓練超過 3 小時）
```bash
kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'
```

## 🎯 主要腳本說明

### train_batch.sh
**在 Pod 內執行的訓練批次腳本**

特點：
- 支持順序訓練多個配置
- 自動日誌記錄
- 失敗時提示是否繼續
- 生成訓練總結報告

修改方式：
```bash
# 編輯 CONFIGS 列表（要訓練的配置）
declare -a CONFIGS=(
  "model_1"
  "model_2"
  "model_3"
)

# 編輯 CONFIG_PARAMS（對應的訓練參數）
declare -A CONFIG_PARAMS=(
  [model_1]="--epochs 20 --batch_size 16"
  [model_2]="--epochs 50 --batch_size 16"
  [model_3]="--epochs 100 --batch_size 8"
)
```

### quick_start_training.sh
**一鍵啟動腳本（最常用）**

執行流程：
1. ✅ 複製訓練腳本到 Pod
2. ✅ 建立 tmux 會話
3. ✅ 啟動訓練
4. ✅ 進入監控模式

使用：
```bash
bash quick_start_training.sh
# 按 Ctrl+B, D 分離（訓練繼續）
# 稍後用此命令重新連接：
kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training
```

### launch_pod_training.sh
**功能完整的管理腳本**

支持選項：
```bash
bash launch_pod_training.sh              # 交互式菜單
bash launch_pod_training.sh --background # 後台啟動
bash launch_pod_training.sh --monitor    # 連接監控
bash launch_pod_training.sh --logs       # 查看日誌
bash launch_pod_training.sh --status     # 檢查狀態
bash launch_pod_training.sh --stop       # 停止訓練
```

### monitor_and_recover.sh
**監控和故障恢復腳本**

在主機端運行（不在 Pod 內）：
```bash
bash k3s/scripts/monitor_and_recover.sh monitor      # 實時監控（5 秒更新）
bash k3s/scripts/monitor_and_recover.sh backup       # 備份訓練數據
bash k3s/scripts/monitor_and_recover.sh auto-save 300  # 自動備份（5 分鐘）
bash k3s/scripts/monitor_and_recover.sh diagnose     # 診斷問題
bash k3s/scripts/monitor_and_recover.sh resume model_name  # 恢復訓練
```

## 🔄 完整工作流

### 開發和測試階段
```bash
# 1. 進入 Pod
kubectl exec -it gpu-pod-working -- bash

# 2. 測試單個訓練
python finetune_encoder.py --model test_exp --epochs 5

# 3. 確認沒問題後，回到主機
exit

# 4. 準備批量訓練配置
vim k3s/scripts/train_batch.sh
```

### 批量訓練階段
```bash
# 1. 快速啟動
cd k3s/scripts
bash quick_start_training.sh

# 2. 進入 tmux 監控模式
# （自動進入，或稍後用 kubectl exec -it gpu-pod-working -- tmux attach-session -t pod_training）

# 3. 監控訓練進度（Ctrl+B, D 分離後在另一個終端）
watch -n 5 'kubectl exec gpu-pod-working -- nvidia-smi'

# 4. 定期備份結果（在第三個終端）
bash k3s/scripts/monitor_and_recover.sh auto-save 300
```

### 訓練完成後
```bash
# 1. 獲取結果
kubectl cp gpu-pod-working:/root/training_results ./local_results

# 2. 查看總結報告
cat local_results/training_summary.txt

# 3. 分析結果
ls -la local_results/checkpoints/
```

## 📊 常用命令速查

### 查看狀態
```bash
# Pod 狀態
kubectl get pods
kubectl describe pod gpu-pod-working

# GPU 使用
kubectl exec gpu-pod-working -- nvidia-smi

# 運行進程
kubectl exec gpu-pod-working -- ps aux | grep python

# 磁盤空間
kubectl exec gpu-pod-working -- df -h
```

### 文件傳輸
```bash
# 主機 → Pod
kubectl cp ./data gpu-pod-working:/root/data

# Pod → 主機
kubectl cp gpu-pod-working:/root/training_results ./results
```

### tmux 快捷鍵（在 Pod 內）
```
Ctrl+B, D       分離會話（訓練繼續）
Ctrl+B, L       切換上一個會話
Ctrl+B, [       進入複製模式
Ctrl+B, :       進入命令模式
Ctrl+C          停止訓練（在 tmux 內）
exit            退出終端
```

### 監控和恢復
```bash
# 實時監控（每 5 秒）
bash k3s/scripts/monitor_and_recover.sh monitor

# 定期備份（每 5 分鐘）
bash k3s/scripts/monitor_and_recover.sh auto-save 300

# 備份一次
bash k3s/scripts/monitor_and_recover.sh backup
```

## ⚠️ 常見問題

### Q：訓練中 SSH 斷開了怎麼辦？
A：訓練在 Pod 的 tmux 內繼續運行。重新連接：
```bash
kubectl exec -it gpu-pod-working -- tmux attach-session -s pod_training
```

### Q：訓練超過 3 小時期限？
A：延長期限：
```bash
kubectl patch pod gpu-pod-working -p '{"spec":{"activeDeadlineSeconds":null}}'
```

### Q：GPU 內存不足？
A：
```bash
# 檢查使用
kubectl exec gpu-pod-working -- nvidia-smi

# 停止訓練
kubectl exec gpu-pod-working -- pkill -f python
```

### Q：訓練失敗了怎麼辦？
A：查看日誌和診斷：
```bash
# 查看日誌
kubectl logs gpu-pod-working

# 診斷問題
bash k3s/scripts/monitor_and_recover.sh diagnose

# 嘗試恢復
bash k3s/scripts/monitor_and_recover.sh resume model_name
```

### Q：如何重啟 Pod？
A：
```bash
# 刪除（自動重啟）
kubectl delete pod gpu-pod-working

# 或手動建立
kubectl apply -f k3s/configs/gpu-pod-working.yaml
```

## 📚 文檔查看

### 快速命令參考
```bash
cat k3s/docs/K3S_GPU_POD_COMMANDS.md
```

### 完整使用指南
```bash
bash k3s/docs/POD_TRAINING_GUIDE.sh
```

### 詳細工作流示例
```bash
bash k3s/examples/training_workflow_examples.sh
```

## 🔧 系統信息

| 項目 | 值 |
|-----|---|
| Kubernetes 版本 | 1.33.6+k3s1 |
| GPU | RTX 5090 (1 顆) |
| GPU 內存 | 32GB |
| Pod 默認期限 | 10800 秒 (3 小時) |
| 圖像 | nvidia/cuda:12.2.0-base-ubuntu22.04 |

## 💡 最佳實踐

1. **在 Pod 內使用 tmux**
   - ✅ Pod 持續運行（不怕斷線）
   - ✅ tmux 保持會話（即使 SSH 斷開）
   - ✅ 隨時重新連接查看進度

2. **定期備份**
   ```bash
   bash k3s/scripts/monitor_and_recover.sh auto-save 300
   ```

3. **監控多個終端**
   - 終端 1：tmux 會話
   - 終端 2：GPU 監控
   - 終端 3：日誌查看

4. **訓練配置分離**
   - 修改 `train_batch.sh` 中的 CONFIGS 列表
   - 使用版本控制追蹤配置變更

## 🚀 下一步

1. 配置你的訓練參數（編輯 `k3s/scripts/train_batch.sh`）
2. 測試單個訓練（進入 Pod 手動執行）
3. 執行快速啟動（`bash quick_start_training.sh`）
4. 監控進度（tmux 或 monitor 腳本）
5. 獲取結果（kubectl cp 複製結果）

## 📞 支援

所有腳本都有詳細注釋。如有問題，查看：
- `k3s/docs/POD_TRAINING_GUIDE.sh` - 交互式指南
- `k3s/docs/K3S_GPU_POD_COMMANDS.md` - 命令參考
- `k3s/examples/training_workflow_examples.sh` - 詳細範例
