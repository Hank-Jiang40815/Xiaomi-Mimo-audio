#!/bin/bash
# 啟動 MiMo-Audio Encoder 微調訓練
# 使用 Docker 容器進行訓練

set -e

# 顏色輸出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MiMo-Audio Encoder Fine-tuning ===${NC}"
echo

# 檢查 Docker 映像是否存在
if ! docker image inspect mimo-audio:latest >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker image 'mimo-audio:latest' not found!${NC}"
    echo "Please build it first with: docker build -t mimo-audio:latest -f Dockerfile ."
    exit 1
fi

# 檢查 split 是否存在
if [ ! -d "data/splits/finetune_optical" ]; then
    echo -e "${YELLOW}⚠️  Training splits not found!${NC}"
    echo "Please create splits first with:"
    echo "  python scripts/data_management/split_selector.py \\"
    echo "    --manifest data/manifests/optical_manifest.json \\"
    echo "    --output-dir data/splits/finetune_optical \\"
    echo "    --mode finetune --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1"
    exit 1
fi

# 預設參數
LORA_RANK=${LORA_RANK:-16}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/optical_lora_r${LORA_RANK}}

echo -e "${GREEN}訓練參數:${NC}"
echo "  • LoRA Rank: $LORA_RANK"
echo "  • LoRA Alpha: $((LORA_RANK * 2))"
echo "  • Epochs: $EPOCHS"
echo "  • Batch Size: $BATCH_SIZE"
echo "  • Learning Rate: $LEARNING_RATE"
echo "  • Output Dir: $OUTPUT_DIR"
echo

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}啟動 Docker 容器進行訓練...${NC}"
echo

# 啟動 Docker 並執行訓練
docker run --gpus all --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --shm-size=16g \
    mimo-audio:latest \
    python finetune_encoder_working.py \
        --data_dir ./data/splits/finetune_optical \
        --tokenizer_path ./models/MiMo-Audio-Tokenizer \
        --output_dir "$OUTPUT_DIR" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha $((LORA_RANK * 2)) \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --device cuda

echo
echo -e "${GREEN}✅ 訓練完成！${NC}"
echo -e "${GREEN}模型已儲存至: $OUTPUT_DIR${NC}"
