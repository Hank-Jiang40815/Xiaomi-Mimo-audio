#!/bin/bash
# Batch ICL Testing on Optical Dataset
# 測試不同 shot 數在 optical 資料集上的效果

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Optical Dataset ICL Testing ===${NC}"
echo

# 檢查 Docker
if ! docker image inspect mimo-audio:latest >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker image 'mimo-audio:latest' not found!${NC}"
    exit 1
fi

# 測試參數
TEST_SPLIT=${TEST_SPLIT:-"data/splits/finetune_optical/test.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/optical_icl_test"}
SHOTS=${SHOTS:-"1 3 5 10"}

echo -e "${GREEN}測試配置:${NC}"
echo "  • 測試集: $TEST_SPLIT"
echo "  • Shot 數: $SHOTS"
echo "  • 輸出目錄: $OUTPUT_DIR"
echo

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# 對每個 shot 數進行測試
for shot in $SHOTS; do
    echo -e "\n${BLUE}=== Testing ${shot}-shot ICL ===${NC}"
    
    shot_output_dir="$OUTPUT_DIR/${shot}shot"
    mkdir -p "$shot_output_dir"
    
    # 執行 ICL 測試
    docker run --gpus all --rm \
        -v "$(pwd)":/workspace \
        -w /workspace \
        mimo-audio:latest \
        python scripts/data_management/test_icl_optical.py \
            --test-split "$TEST_SPLIT" \
            --shots "$shot" \
            --output-dir "$shot_output_dir" \
            --num-samples 20
    
    echo -e "${GREEN}✅ ${shot}-shot 測試完成${NC}"
done

echo
echo -e "${GREEN}=== 所有測試完成！===${NC}"
echo -e "結果已儲存至: $OUTPUT_DIR"
echo
echo "查看結果:"
echo "  ls -lh $OUTPUT_DIR/*/"
