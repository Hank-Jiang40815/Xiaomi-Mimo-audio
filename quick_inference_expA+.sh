#!/bin/bash
# 使用 exp(A+) 配置快速推論新音檔
# 基於 commit f56da64 的實驗配置
# 在 Docker 環境中執行

set -e

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Exp(A+) 快速推論工具 - 10字限制配置                       ║"
echo "║     (Docker 模式)                                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ 錯誤: 找不到 Docker${NC}"
    echo "請先安裝 Docker"
    exit 1
fi

# 檢查 Docker image
if ! docker images | grep -q "mimo-audio"; then
    echo -e "${RED}❌ 錯誤: 找不到 mimo-audio:latest Docker image${NC}"
    echo "請先建立 Docker image:"
    echo "  docker build -t mimo-audio:latest ."
    exit 1
fi

# 檢查參數
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}使用方式:${NC}"
    echo "  $0 <input_audio.wav> [output_audio.wav]"
    echo ""
    echo -e "${YELLOW}範例:${NC}"
    echo "  $0 my_noisy_audio.wav"
    echo "  $0 my_noisy_audio.wav enhanced_output.wav"
    echo "  $0 examples/ldv/mix/boy1_papercup_LDV_041.wav"
    echo ""
    echo -e "${YELLOW}配置說明:${NC}"
    echo "  • 40-shot In-Context Learning"
    echo "  • Instruction: 10 字中文限制"
    echo "  • Dataset: optical_denoised (001-040)"
    echo "  • Model: MiMo-Audio-7B-Base"
    echo "  • 執行環境: Docker (mimo-audio:latest)"
    echo ""
    exit 1
fi

INPUT_AUDIO="$1"
OUTPUT_AUDIO="${2:-outputs/enhanced_$(basename "$INPUT_AUDIO")}"

# 檢查輸入檔案
if [ ! -f "$INPUT_AUDIO" ]; then
    echo -e "${RED}❌ 錯誤: 找不到輸入檔案: $INPUT_AUDIO${NC}"
    exit 1
fi

# 建立輸出目錄
mkdir -p "$(dirname "$OUTPUT_AUDIO")"

# 轉換為容器內路徑
# 假設檔案在 examples/ 或當前目錄，需要對應到 /app/ 底下
CONTAINER_INPUT="$INPUT_AUDIO"
CONTAINER_OUTPUT="$OUTPUT_AUDIO"

# 如果是絕對路徑，轉換為相對路徑（Docker 會掛載當前目錄到 /app）
if [[ "$INPUT_AUDIO" == /* ]]; then
    # 取得相對於當前目錄的路徑
    REL_INPUT=$(realpath --relative-to="$(pwd)" "$INPUT_AUDIO")
    CONTAINER_INPUT="/app/$REL_INPUT"
fi

if [[ "$OUTPUT_AUDIO" == /* ]]; then
    REL_OUTPUT=$(realpath --relative-to="$(pwd)" "$OUTPUT_AUDIO")
    CONTAINER_OUTPUT="/app/$REL_OUTPUT"
else
    CONTAINER_OUTPUT="/app/$OUTPUT_AUDIO"
fi

echo -e "${GREEN}✓${NC} 輸入檔案: ${BLUE}$INPUT_AUDIO${NC}"
echo -e "${GREEN}✓${NC} 輸出檔案: ${BLUE}$OUTPUT_AUDIO${NC}"
echo -e "${GREEN}✓${NC} Docker 輸入: ${BLUE}$CONTAINER_INPUT${NC}"
echo -e "${GREEN}✓${NC} Docker 輸出: ${BLUE}$CONTAINER_OUTPUT${NC}"
echo ""

# 建立臨時 Python 腳本（在當前目錄，讓 Docker 能存取）
TEMP_SCRIPT="./temp_inference_$(date +%s).py"

cat > "$TEMP_SCRIPT" << 'PYTHON_SCRIPT'
import sys
import os
from src.mimo_audio.mimo_audio import MimoAudio

# 從命令列參數取得檔案路徑
input_audio = sys.argv[1]
output_audio = sys.argv[2]

print("=" * 60)
print("🚀 載入模型...")
print("=" * 60)

# Load model (直接初始化，不是 from_pretrained)
model = MimoAudio(
    model_path="models/MiMo-Audio-7B-Base",
    mimo_audio_tokenizer_path="models/MiMo-Audio-Tokenizer"
)

print("\n✅ 模型載入完成\n")

# Exp(A+) 配置: 10 字限制
instruction = "Enhance the audio quality and remove noise from the input speech. IMPORTANT: You MUST preserve the exact original speech content and transcript. The speech content is exactly 10 Chinese characters. Only improve the audio quality, do not change any words."

# 40-shot examples (001-040)
prompt_examples = []
for i in range(1, 41):
    prompt_examples.append({
        "input_audio": f"examples/optical_denoised/mix/boy1_WOLDVlean_{i:03d}.wav",
        "output_audio": f"examples/optical_denoised/spk1/boy1_papercup_clean_{i:03d}.wav",
        "output_transcription": "",  # 不需要轉錄文字
    })

print("=" * 60)
print("🎯 開始推論")
print("=" * 60)
print(f"配置: Exp(A+) - 10 字中文限制")
print(f"Shot count: {len(prompt_examples)}")
print(f"輸入: {input_audio}")
print(f"輸出: {output_audio}")
print("=" * 60 + "\n")

try:
    text_output = model.in_context_learning_s2s(
        instruction, 
        prompt_examples, 
        input_audio, 
        max_new_tokens=8192, 
        output_audio_path=output_audio
    )
    
    print("\n" + "=" * 60)
    print("✅ 推論成功完成！")
    print("=" * 60)
    print(f"Text output: {text_output}")
    print(f"Enhanced audio saved to: {output_audio}")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 推論失敗！")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

# 執行推論（在 Docker 中）
echo -e "${YELLOW}⏳ 在 Docker 容器中執行推論...${NC}"
echo ""

# 使用 Docker 執行，掛載整個當前目錄
docker run --gpus all --rm \
  -v "$(pwd)":/app \
  -w /app \
  mimo-audio:latest \
  python "$TEMP_SCRIPT" "$CONTAINER_INPUT" "$CONTAINER_OUTPUT"

# 清理臨時檔案
rm -f "$TEMP_SCRIPT"

# 顯示結果
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     推論完成                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$OUTPUT_AUDIO" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_AUDIO" | awk '{print $5}')
    echo -e "${GREEN}✅ 成功生成增強音檔${NC}"
    echo -e "   檔案: ${BLUE}$OUTPUT_AUDIO${NC}"
    echo -e "   大小: ${FILE_SIZE}"
    echo ""
    
    # 如果有 ffmpeg，顯示音訊資訊
    if command -v ffprobe &> /dev/null; then
        echo -e "${YELLOW}📊 音訊資訊:${NC}"
        ffprobe -v quiet -show_entries format=duration,bit_rate -of default=noprint_wrappers=1 "$OUTPUT_AUDIO" 2>/dev/null | sed 's/^/   /'
        echo ""
    fi
    
    echo -e "${YELLOW}💡 播放指令:${NC}"
    echo "   ffplay -autoexit -nodisp $OUTPUT_AUDIO"
    echo ""
else
    echo -e "${RED}❌ 推論失敗：未生成輸出檔案${NC}"
    exit 1
fi
