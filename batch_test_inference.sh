#!/bin/bash
# 批次測試微調後的 Encoder

CHECKPOINT="outputs/optical_lora_r32_e100/best_model.pt"
TOKENIZER="models/MiMo-Audio-Tokenizer"
OUTPUT_DIR="outputs/batch_test_inference"

# 測試的音檔列表（選擇不同的樣本）
TEST_FILES=(
    "examples/optical/mix/boy1_WOLDV_001.wav"
    "examples/optical/mix/boy1_WOLDV_010.wav"
    "examples/optical/mix/boy1_WOLDV_050.wav"
    "examples/optical/mix/boy1_WOLDV_100.wav"
    "examples/optical/mix/boy1_WOLDV_200.wav"
)

echo "=========================================="
echo "批次測試微調後的 Encoder Inference"
echo "=========================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "測試樣本數: ${#TEST_FILES[@]}"
echo ""

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# 測試每個檔案
SUCCESS=0
FAILED=0

for INPUT in "${TEST_FILES[@]}"; do
    if [ ! -f "$INPUT" ]; then
        echo "⚠️  跳過: $INPUT (檔案不存在)"
        ((FAILED++))
        continue
    fi
    
    BASENAME=$(basename "$INPUT" .wav)
    OUTPUT="$OUTPUT_DIR/enhanced_${BASENAME}.wav"
    
    echo "────────────────────────────────────────"
    echo "🎵 測試: $BASENAME"
    echo "────────────────────────────────────────"
    
    docker run --gpus all --rm \
        -v "$(pwd)":/workspace -w /workspace \
        mimo-audio:latest python test_finetuned_inference.py \
        --checkpoint "$CHECKPOINT" \
        --tokenizer-path "$TOKENIZER" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --device cuda 2>&1 | tail -10
    
    if [ $? -eq 0 ] && [ -f "$OUTPUT" ]; then
        SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
        echo "✅ 成功: $OUTPUT ($SIZE)"
        ((SUCCESS++))
    else
        echo "❌ 失敗: $BASENAME"
        ((FAILED++))
    fi
    echo ""
done

echo "=========================================="
echo "批次測試完成"
echo "=========================================="
echo ""
echo "📊 結果統計:"
echo "   成功: $SUCCESS / ${#TEST_FILES[@]}"
echo "   失敗: $FAILED / ${#TEST_FILES[@]}"
echo ""
echo "📁 輸出目錄: $OUTPUT_DIR"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "🎧 播放增強後的音檔:"
    ls -1 "$OUTPUT_DIR"/*.wav 2>/dev/null | head -5 | while read file; do
        echo "   ffplay -autoexit -nodisp $file"
    done
fi
