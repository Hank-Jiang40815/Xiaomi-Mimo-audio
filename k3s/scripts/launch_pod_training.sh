#!/bin/bash

# ============================================
# K3S GPU Pod + tmux 訓練一鍵啟動腳本
# ============================================
# 用途：從主機直接啟動 Pod 內的 tmux 訓練會話
# 用法：bash launch_pod_training.sh [config_name]

set -e

# ============================================
# 配置
# ============================================

POD_NAME="gpu-pod-working"
TMUX_SESSION="pod_training"
PROJECT_ROOT="/root"
TRAINING_SCRIPT="train_batch.sh"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 函數定義
# ============================================

log() {
  echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
  echo -e "${GREEN}✅ $1${NC}"
}

error() {
  echo -e "${RED}❌ $1${NC}"
}

warning() {
  echo -e "${YELLOW}⚠️ $1${NC}"
}

# ============================================
# 前置檢查
# ============================================

check_prerequisites() {
  log "執行前置檢查..."
  
  # 檢查 kubectl
  if ! command -v kubectl &> /dev/null; then
    error "kubectl 未安裝"
    exit 1
  fi
  
  # 檢查 Pod 是否運行
  if ! kubectl get pod "$POD_NAME" &> /dev/null; then
    error "Pod '$POD_NAME' 不存在"
    exit 1
  fi
  
  POD_STATUS=$(kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}')
  if [ "$POD_STATUS" != "Running" ]; then
    error "Pod 狀態不是 Running，當前狀態：$POD_STATUS"
    exit 1
  fi
  
  success "前置檢查完成"
}

# ============================================
# 檢查 Pod 內的必要文件
# ============================================

check_pod_files() {
  log "檢查 Pod 內的文件..."
  
  # 檢查是否有訓練腳本
  if kubectl exec "$POD_NAME" -- test -f "$PROJECT_ROOT/$TRAINING_SCRIPT" 2>/dev/null; then
    success "訓練腳本已存在"
  else
    warning "訓練腳本不存在，將復製..."
    copy_files_to_pod
  fi
}

# ============================================
# 複製文件到 Pod
# ============================================

copy_files_to_pod() {
  log "複製文件到 Pod..."
  
  # 複製訓練腳本
  if [ -f "./train_batch.sh" ]; then
    kubectl cp ./train_batch.sh "$POD_NAME:$PROJECT_ROOT/"
    success "train_batch.sh 已複製"
  else
    warning "train_batch.sh 不存在於當前目錄"
  fi
  
  # 複製訓練代碼（如果需要）
  if [ -d "./src" ]; then
    kubectl cp ./src "$POD_NAME:$PROJECT_ROOT/"
    success "src 目錄已複製"
  fi
  
  # 複製數據（如果需要）
  if [ -d "./data" ] && [ -n "$COPY_DATA" ]; then
    log "複製數據目錄（可能需要時間）..."
    kubectl cp ./data "$POD_NAME:$PROJECT_ROOT/"
    success "data 目錄已複製"
  fi
}

# ============================================
# 在 Pod 內設置 tmux
# ============================================

setup_pod_tmux() {
  log "在 Pod 內設置 tmux 會話..."
  
  # 檢查 tmux 是否已安裝
  kubectl exec "$POD_NAME" -- which tmux > /dev/null 2>&1 || {
    warning "Pod 內未安裝 tmux，嘗試安裝..."
    kubectl exec "$POD_NAME" -- apt-get update > /dev/null && \
    kubectl exec "$POD_NAME" -- apt-get install -y tmux > /dev/null 2>&1 && \
    success "tmux 安裝完成" || \
    warning "tmux 安裝失敗，將使用其他方式運行"
  }
  
  # 檢查會話是否已存在
  if kubectl exec "$POD_NAME" -- tmux list-sessions 2>/dev/null | grep -q "$TMUX_SESSION"; then
    warning "tmux 會話 '$TMUX_SESSION' 已存在"
    log "將使用現有會話"
  else
    log "建立新的 tmux 會話..."
    kubectl exec "$POD_NAME" -- tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT"
    success "tmux 會話已建立"
  fi
}

# ============================================
# 啟動訓練
# ============================================

start_training() {
  local config_name=$1
  
  log "在 tmux 會話內啟動訓練..."
  
  if [ -z "$config_name" ]; then
    # 沒有指定配置，運行完整批次
    kubectl exec "$POD_NAME" -- tmux send-keys -t "$TMUX_SESSION" \
      "cd $PROJECT_ROOT && bash $TRAINING_SCRIPT" Enter
    success "批量訓練已啟動"
  else
    # 指定配置，運行單個訓練
    warning "此功能需要修改 train_batch.sh 以支持單配置模式"
    kubectl exec "$POD_NAME" -- tmux send-keys -t "$TMUX_SESSION" \
      "cd $PROJECT_ROOT && bash $TRAINING_SCRIPT $config_name" Enter
  fi
}

# ============================================
# 監控訓練
# ============================================

monitor_training() {
  log "連接到 tmux 會話進行監控..."
  echo ""
  echo "════════════════════════════════════════"
  echo "進入 Pod 的 tmux 會話"
  echo "════════════════════════════════════════"
  echo "按以下鍵盤快捷鍵："
  echo "  Ctrl+B, D  - 分離會話（訓練繼續）"
  echo "  Ctrl+B, [  - 進入複製模式（查看歷史）"
  echo "  Ctrl+C     - 停止訓練"
  echo "════════════════════════════════════════"
  echo ""
  
  kubectl exec -it "$POD_NAME" -- tmux attach-session -t "$TMUX_SESSION"
}

# ============================================
# 後台啟動（不進入交互式）
# ============================================

start_background() {
  log "以後台模式啟動訓練（不進入交互式會話）"
  
  kubectl exec "$POD_NAME" -- tmux send-keys -t "$TMUX_SESSION" \
    "cd $PROJECT_ROOT && bash $TRAINING_SCRIPT" Enter
  
  success "訓練已在後台啟動"
  success "使用以下命令監控進度："
  echo "  kubectl exec -it $POD_NAME -- tmux attach-session -t $TMUX_SESSION"
}

# ============================================
# 查看訓練日誌
# ============================================

show_logs() {
  log "顯示訓練日誌..."
  
  kubectl logs -f "$POD_NAME" | tail -50
}

# ============================================
# 顯示主菜單
# ============================================

show_menu() {
  echo ""
  echo "════════════════════════════════════════"
  echo "K3S GPU Pod + tmux 訓練管理"
  echo "════════════════════════════════════════"
  echo "1. 啟動訓練並進入監控"
  echo "2. 啟動訓練到後台"
  echo "3. 連接到已運行的會話"
  echo "4. 查看訓練日誌"
  echo "5. 檢查 Pod 狀態"
  echo "6. 複製文件到 Pod"
  echo "7. 停止訓練"
  echo "8. 退出"
  echo "════════════════════════════════════════"
  echo ""
}

# ============================================
# 檢查狀態
# ============================================

show_status() {
  log "檢查 Pod 狀態..."
  echo ""
  
  kubectl get pod "$POD_NAME" -o wide
  echo ""
  
  log "檢查 GPU..."
  kubectl exec "$POD_NAME" -- nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "無法獲取 GPU 信息"
  echo ""
  
  log "檢查 tmux 會話..."
  kubectl exec "$POD_NAME" -- tmux list-sessions 2>/dev/null || echo "沒有 tmux 會話"
  echo ""
  
  log "檢查訓練進程..."
  kubectl exec "$POD_NAME" -- ps aux | grep -E "python|train" | grep -v grep || echo "沒有訓練進程"
  echo ""
}

# ============================================
# 停止訓練
# ============================================

stop_training() {
  warning "確定要停止訓練嗎？(y/n)"
  read -r confirm
  
  if [ "$confirm" = "y" ]; then
    log "停止訓練進程..."
    kubectl exec "$POD_NAME" -- pkill -f "python.*train" || true
    success "訓練已停止"
  else
    log "已取消"
  fi
}

# ============================================
# 主程序
# ============================================

main() {
  log "K3S GPU Pod + tmux 訓練啟動器"
  
  # 檢查命令行參數
  case "${1:-}" in
    --background|-b)
      check_prerequisites
      check_pod_files
      setup_pod_tmux
      start_background
      exit 0
      ;;
    --monitor|-m)
      check_prerequisites
      monitor_training
      exit 0
      ;;
    --logs|-l)
      check_prerequisites
      show_logs
      exit 0
      ;;
    --status|-s)
      check_prerequisites
      show_status
      exit 0
      ;;
    --help|-h)
      echo "使用方法："
      echo "  bash $0                 - 交互式菜單"
      echo "  bash $0 --background    - 後台啟動訓練"
      echo "  bash $0 --monitor       - 連接到監控"
      echo "  bash $0 --logs          - 查看日誌"
      echo "  bash $0 --status        - 檢查狀態"
      echo "  bash $0 --stop          - 停止訓練"
      exit 0
      ;;
    --stop)
      check_prerequisites
      stop_training
      exit 0
      ;;
    *)
      # 交互式菜單
      check_prerequisites
      check_pod_files
      setup_pod_tmux
      
      while true; do
        show_menu
        read -p "請選擇 (1-8): " choice
        
        case $choice in
          1)
            start_training
            monitor_training
            ;;
          2)
            start_background
            ;;
          3)
            monitor_training
            ;;
          4)
            show_logs
            ;;
          5)
            show_status
            ;;
          6)
            copy_files_to_pod
            ;;
          7)
            stop_training
            ;;
          8)
            success "退出程序"
            exit 0
            ;;
          *)
            error "無效選擇"
            ;;
        esac
        
        echo ""
        read -p "按 Enter 繼續..."
      done
      ;;
  esac
}

# ============================================
# 執行主程序
# ============================================

main "$@"
