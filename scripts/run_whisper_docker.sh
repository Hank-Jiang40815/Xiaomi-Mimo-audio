#!/bin/bash
# Whisper Docker 轉錄腳本
# 使用 Whisper Docker 容器進行音訊轉錄

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎤 Whisper Docker 轉錄工具${NC}"
echo "================================"
echo ""

# 檢查 Docker 映像是否存在
if ! docker images | grep -q "whisper-asr"; then
    echo -e "${YELLOW}⚠️  Whisper Docker 映像不存在，開始構建...${NC}"
    docker build -f Dockerfile.whisper -t whisper-asr:latest .
fi

# 設定預設參數
AUDIO_DIR="${1:-examples/optical/spk1}"
OUTPUT_JSON="${2:-data/transcriptions/optical_transcriptions.json}"
MODEL_SIZE="${3:-base}"
LANGUAGE="${4:-zh}"

echo "📂 音檔目錄: $AUDIO_DIR"
echo "💾 輸出檔案: $OUTPUT_JSON"
echo "🤖 模型大小: $MODEL_SIZE"
echo "🌏 語言: $LANGUAGE"
echo ""

# 執行轉錄
echo -e "${GREEN}開始轉錄...${NC}"
docker run --rm \
    --gpus all \
    -v "$(pwd):/workspace" \
    -w /workspace \
    whisper-asr:latest \
    python scripts/data_management/transcribe_optical.py \
        --audio-dir "$AUDIO_DIR" \
        --output "$OUTPUT_JSON" \
        --model-size "$MODEL_SIZE" \
        --language "$LANGUAGE"

echo ""
echo -e "${GREEN}✅ 轉錄完成！${NC}"
