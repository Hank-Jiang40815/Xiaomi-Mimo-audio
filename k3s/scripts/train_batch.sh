#!/bin/bash

# ============================================
# 連續訓練管理腳本（實際使用版本）
# ============================================
# 使用方法：
#   1. 編輯下方的訓練配置
#   2. chmod +x train_batch.sh
#   3. 進入 Pod: kubectl exec -it gpu-pod-working -- bash
#   4. 在 Pod 內執行: bash train_batch.sh

set -e  # 錯誤時停止

# ============================================
# 配置區域
# ============================================

# 訓練配置列表
declare -a CONFIGS=(
  "exp_a_r32_e20"
  "exp_a_r32_e50"
  "exp_b_r32_e20"
  "exp_b_r32_e50"
  "exp_c_r32_e100"
)

# 對應的參數
declare -A CONFIG_PARAMS=(
  [exp_a_r32_e20]="--model exp_a --rank 32 --epochs 20 --batch_size 16"
  [exp_a_r32_e50]="--model exp_a --rank 32 --epochs 50 --batch_size 16"
  [exp_b_r32_e20]="--model exp_b --rank 32 --epochs 20 --batch_size 16"
  [exp_b_r32_e50]="--model exp_b --rank 32 --epochs 50 --batch_size 16"
  [exp_c_r32_e100]="--model exp_c --rank 32 --epochs 100 --batch_size 8"
)

# 輸出目錄
OUTPUT_BASE="/root/training_results"
LOGS_DIR="$OUTPUT_BASE/logs"
CHECKPOINTS_DIR="$OUTPUT_BASE/checkpoints"

# 建立目錄
mkdir -p "$LOGS_DIR" "$CHECKPOINTS_DIR"

# ============================================
# 日誌函數
# ============================================

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_success() {
  echo "✅ [$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
  echo "❌ [$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================
# 前置檢查
# ============================================

log "開始前置檢查..."

# 檢查 GPU 可用性
if ! nvidia-smi > /dev/null 2>&1; then
  log_error "GPU 不可用！"
  exit 1
fi

# 顯示 GPU 信息
log "GPU 狀態："
nvidia-smi | head -n 5

# 檢查是否有足夠的磁盤空間
DISK_USAGE=$(df /root | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
  log_error "磁盤使用超過 80%！請清理空間"
  exit 1
fi

log_success "前置檢查完成"

# ============================================
# 訓練循環
# ============================================

TOTAL=${#CONFIGS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CONFIGS=()

log "準備訓練 $TOTAL 個配置"

for i in "${!CONFIGS[@]}"; do
  CONFIG=${CONFIGS[$i]}
  CURRENT=$((i + 1))
  
  log "════════════════════════════════════════"
  log "【$CURRENT/$TOTAL】開始訓練：$CONFIG"
  log "════════════════════════════════════════"
  
  # 提取參數
  PARAMS=${CONFIG_PARAMS[$CONFIG]}
  LOG_FILE="$LOGS_DIR/${CONFIG}_training.log"
  CHECKPOINT_DIR="$CHECKPOINTS_DIR/$CONFIG"
  
  mkdir -p "$CHECKPOINT_DIR"
  
  # 記錄開始時間
  START_TIME=$(date +%s)
  
  # 執行訓練
  log "執行命令："
  log "python finetune_encoder.py $PARAMS --output $CHECKPOINT_DIR"
  
  if python finetune_encoder.py $PARAMS --output "$CHECKPOINT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    # 訓練成功
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_success "$CONFIG 訓練完成 (耗時: $((DURATION / 60)) 分 $((DURATION % 60)) 秒)"
    ((SUCCESS_COUNT++))
    
    # 運行評估（可選）
    if [ -f "evaluate.py" ]; then
      log "運行評估..."
      python evaluate.py --checkpoint "$CHECKPOINT_DIR" | tee -a "$LOGS_DIR/${CONFIG}_eval.log"
    fi
    
    # 保存訓練成功的標記
    echo "success" > "$CHECKPOINT_DIR/status.txt"
    
  else
    # 訓練失敗
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_error "$CONFIG 訓練失敗 (耗時: $((DURATION / 60)) 分 $((DURATION % 60)) 秒)"
    ((FAIL_COUNT++))
    FAILED_CONFIGS+=("$CONFIG")
    
    # 保存失敗的標記
    echo "failed" > "$CHECKPOINT_DIR/status.txt"
    
    # 詢問是否繼續
    log "是否繼續下一個訓練？(y/n)"
    read -r CONTINUE
    if [ "$CONTINUE" != "y" ]; then
      log "用戶選擇中止訓練"
      break
    fi
  fi
  
  # 在訓練之間等待，讓 GPU 冷卻
  if [ $((i + 1)) -lt $TOTAL ]; then
    log "等待 30 秒讓 GPU 冷卻..."
    sleep 30
  fi
  
  # 顯示 GPU 狀態
  log "GPU 當前狀態："
  nvidia-smi | grep "Processes\|No running"
done

# ============================================
# 訓練完成總結
# ============================================

log "════════════════════════════════════════"
log "訓練批次完成！"
log "════════════════════════════════════════"

echo ""
echo "📊 訓練結果統計："
echo "  ✅ 成功: $SUCCESS_COUNT/$TOTAL"
echo "  ❌ 失敗: $FAIL_COUNT/$TOTAL"

if [ $FAIL_COUNT -gt 0 ]; then
  echo ""
  echo "失敗的配置："
  for config in "${FAILED_CONFIGS[@]}"; do
    echo "  - $config"
  done
fi

# 生成總結報告
cat > "$OUTPUT_BASE/training_summary.txt" << EOF
訓練批次執行報告
================================
執行時間: $(date)
總配置數: $TOTAL
成功: $SUCCESS_COUNT
失敗: $FAIL_COUNT

配置詳情：
$(for config in "${CONFIGS[@]}"; do
  status_file="$CHECKPOINTS_DIR/$config/status.txt"
  if [ -f "$status_file" ]; then
    status=$(cat "$status_file")
    echo "- $config: $status"
  fi
done)

日誌位置: $LOGS_DIR
檢查點位置: $CHECKPOINTS_DIR
EOF

log_success "報告已保存: $OUTPUT_BASE/training_summary.txt"

# 列出所有生成的文件
log "生成的文件："
find "$OUTPUT_BASE" -type f -exec ls -lh {} \; | awk '{print "  " $9 " (" $5 ")"}'

echo ""
log_success "所有訓練完成！"
