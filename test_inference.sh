#!/bin/bash
# 在 Docker 中測試微調後的 Encoder Inference

set -e

echo "=========================================="
echo "🧪 測試微調後 Encoder Inference"
echo "=========================================="
echo ""

# 預設參數
CHECKPOINT="${CHECKPOINT:-outputs/optical_lora_r32_e100/checkpoint_epoch_2.pt}"
INPUT="${INPUT:-examples/optical/mix/boy1_WOLDV_050.wav}"
OUTPUT="${OUTPUT:-outputs/test_inference/enhanced_epoch2.wav}"

echo "配置:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT"
echo ""

# 檢查檔案
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "❌ 輸入檔案不存在: $INPUT"
    exit 1
fi

# 建立輸出目錄
mkdir -p "$(dirname "$OUTPUT")"

echo "🐳 在 Docker 中執行..."
echo ""

# 執行 Docker
docker run --gpus all --rm \
    -v "$(pwd)":/workspace \
    -w /workspace \
    mimo-audio:latest \
    python test_finetuned_inference.py \
        --checkpoint "$CHECKPOINT" \
        --tokenizer-path ./models/MiMo-Audio-Tokenizer \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --device cuda

echo ""
echo "=========================================="
echo "✅ 測試完成！"
echo "=========================================="
echo ""

if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
    echo "📁 輸出檔案: $OUTPUT"
    echo "📊 大小: $FILE_SIZE"
    echo ""
    echo "🎧 播放指令:"
    echo "   原始: ffplay -autoexit -nodisp $INPUT"
    echo "   增強: ffplay -autoexit -nodisp $OUTPUT"
else
    echo "❌ 輸出檔案未生成"
    exit 1
fi
