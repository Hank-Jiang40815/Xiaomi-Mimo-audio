#!/bin/bash

# ============================================
# 訓練監控和恢復腳本
# ============================================
# 在主機端執行（不在 Pod 內）
# 用途：監控訓練進度、保存日誌、自動恢復

POD_NAME="gpu-pod-working"
MONITOR_LOG="/tmp/training_monitor.log"

# ============================================
# 即時監控（每 5 秒更新一次）
# ============================================

monitor_training() {
  echo "=== 開始監控訓練 ==="
  echo "按 Ctrl+C 停止監控"
  echo ""
  
  iteration=0
  while true; do
    clear
    ((iteration++))
    
    echo "════════════════════════════════════════"
    echo "訓練監控 #$iteration ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "════════════════════════════════════════"
    echo ""
    
    # Pod 狀態
    echo "【Pod 狀態】"
    kubectl get pod "$POD_NAME" --no-headers
    echo ""
    
    # GPU 使用情況
    echo "【GPU 使用】"
    kubectl exec "$POD_NAME" -- nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory --format=csv,noheader 2>/dev/null || echo "無法獲取 GPU 信息"
    echo ""
    
    # CPU 和內存使用
    echo "【系統資源】"
    kubectl exec "$POD_NAME" -- free -h 2>/dev/null | head -2 || echo "無法獲取內存信息"
    echo ""
    
    # 訓練進程
    echo "【訓練進程】"
    kubectl exec "$POD_NAME" -- ps aux 2>/dev/null | grep -E "python|train" | grep -v grep || echo "沒有訓練進程"
    echo ""
    
    # 最新的訓練日誌（最後 3 行）
    echo "【訓練日誌（最後 3 行）】"
    kubectl logs "$POD_NAME" --tail=3 2>/dev/null || echo "無可用日誌"
    echo ""
    
    # 檢查 Pod 健康狀態
    echo "【健康狀態檢查】"
    if kubectl get pod "$POD_NAME" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q "True"; then
      echo "✅ Pod 運行正常"
    else
      echo "⚠️ Pod 可能有問題"
      kubectl describe pod "$POD_NAME" | grep -A 5 "Events:"
    fi
    
    echo ""
    echo "下一次更新在 5 秒後..."
    sleep 5
  done
}

# ============================================
# 保存訓練數據
# ============================================

backup_training_data() {
  POD_NAME=$1
  BACKUP_DIR="/tmp/training_backup_$(date +%s)"
  
  echo "備份訓練數據到：$BACKUP_DIR"
  mkdir -p "$BACKUP_DIR"
  
  # 從 Pod 複製訓練結果
  kubectl cp "$POD_NAME:/root/training_results" "$BACKUP_DIR/" 2>/dev/null && \
    echo "✅ 訓練結果已備份" || echo "⚠️ 無法備份訓練結果"
  
  # 從 Pod 複製日誌
  kubectl cp "$POD_NAME:/root/training/logs" "$BACKUP_DIR/" 2>/dev/null && \
    echo "✅ 日誌已備份" || echo "⚠️ 無法備份日誌"
  
  echo "備份位置：$BACKUP_DIR"
}

# ============================================
# 自動恢復失敗的訓練
# ============================================

resume_failed_training() {
  POD_NAME=$1
  CONFIG=$2
  
  echo "嘗試恢復訓練：$CONFIG"
  
  # 檢查檢查點是否存在
  if kubectl exec "$POD_NAME" -- ls "/root/training_results/checkpoints/$CONFIG/" > /dev/null 2>&1; then
    echo "找到檢查點，嘗試從檢查點恢復..."
    
    # 在 Pod 內執行恢復命令
    kubectl exec "$POD_NAME" -- bash -c "
      cd /root
      python finetune_encoder.py \
        --model $CONFIG \
        --resume /root/training_results/checkpoints/$CONFIG \
        --epochs 100 \
        2>&1 | tee -a /root/training_results/logs/${CONFIG}_resume.log
    "
  else
    echo "未找到檢查點，重新開始訓練..."
    kubectl exec "$POD_NAME" -- bash -c "
      cd /root
      python finetune_encoder.py \
        --model $CONFIG \
        --epochs 100 \
        2>&1 | tee -a /root/training_results/logs/${CONFIG}_restart.log
    "
  fi
}

# ============================================
# 診斷 Pod 問題
# ============================================

diagnose_pod() {
  POD_NAME=$1
  
  echo "════════════════════════════════════════"
  echo "Pod 診斷報告"
  echo "════════════════════════════════════════"
  echo ""
  
  # 基本信息
  echo "【基本信息】"
  kubectl describe pod "$POD_NAME" | head -20
  echo ""
  
  # 事件日誌
  echo "【最近事件】"
  kubectl describe pod "$POD_NAME" | tail -20
  echo ""
  
  # 完整日誌
  echo "【容器日誌】"
  kubectl logs "$POD_NAME" --all-containers=true
  echo ""
  
  # 檢查磁盤空間
  echo "【磁盤使用情況】"
  kubectl exec "$POD_NAME" -- df -h
  echo ""
  
  # GPU 內存
  echo "【GPU 內存】"
  kubectl exec "$POD_NAME" -- nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
  echo ""
}

# ============================================
# 設置自動保存
# ============================================

setup_auto_save() {
  POD_NAME=$1
  INTERVAL=$2  # 秒數
  
  echo "設置自動保存（每 $INTERVAL 秒保存一次）"
  
  # 在主機端建立監視進程
  while true; do
    backup_training_data "$POD_NAME"
    echo "自動保存完成於 $(date)"
    sleep "$INTERVAL"
  done
}

# ============================================
# 主程序
# ============================================

case "${1:-monitor}" in
  monitor)
    monitor_training
    ;;
  backup)
    backup_training_data "$POD_NAME"
    ;;
  resume)
    resume_failed_training "$POD_NAME" "$2"
    ;;
  diagnose)
    diagnose_pod "$POD_NAME"
    ;;
  auto-save)
    INTERVAL=${2:-300}  # 預設 5 分鐘
    setup_auto_save "$POD_NAME" "$INTERVAL"
    ;;
  *)
    echo "用法："
    echo "  $0 monitor          - 即時監控訓練"
    echo "  $0 backup           - 備份訓練數據"
    echo "  $0 resume <config>  - 恢復失敗的訓練"
    echo "  $0 diagnose         - 診斷 Pod 問題"
    echo "  $0 auto-save [sec]  - 自動定期保存數據"
    echo ""
    echo "示例："
    echo "  $0 monitor"
    echo "  $0 auto-save 300    # 每 5 分鐘保存一次"
    exit 1
    ;;
esac
