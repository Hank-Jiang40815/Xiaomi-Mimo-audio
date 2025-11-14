#!/bin/bash
# Exp(A++) 增強版推論工具 - 支援可變 shot 數量和自訂 shot 選擇
# 基於 Exp(A+) 升級
# 在 Docker 環境中執行

set -e

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 預設值
SHOTS=40
DATASET="optical_denoised"
INSTRUCTION="10-char"  # 10-char, none, custom
CUSTOM_INSTRUCTION=""
SHOT_RANGE=""  # 例如 "1-10,15,20-25"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Exp(A++) 增強版推論工具                                    ║"
echo "║     • 可變 shot 數量                                           ║"
echo "║     • 自訂 shot 選擇                                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 顯示使用說明
show_usage() {
    echo -e "${YELLOW}使用方式:${NC}"
    echo "  $0 <input_audio.wav> [options]"
    echo ""
    echo -e "${YELLOW}必要參數:${NC}"
    echo "  input_audio.wav           輸入音檔路徑"
    echo ""
    echo -e "${YELLOW}可選參數:${NC}"
    echo "  -o, --output FILE         輸出音檔路徑 (預設: outputs/enhanced_*.wav)"
    echo "  -s, --shots N             Shot 數量 (預設: 40)"
    echo "  -r, --range RANGE         自訂 shot 範圍 (例如: '1-10,15,20-25')"
    echo "  -d, --dataset NAME        資料集 (optical_denoised, optical, ldv)"
    echo "  -i, --instruction TYPE    Instruction 類型:"
    echo "                              10-char  - 10字中文限制 (預設)"
    echo "                              none     - 不限制"
    echo "                              custom   - 自訂 (需搭配 --custom-inst)"
    echo "  --custom-inst TEXT        自訂 instruction 內容"
    echo "  --max-tokens N            最大生成 token 數 (預設: 8192)"
    echo "  -h, --help                顯示此說明"
    echo ""
    echo -e "${YELLOW}範例:${NC}"
    echo "  # 使用預設 40-shot"
    echo "  $0 my_audio.wav"
    echo ""
    echo "  # 使用 10-shot"
    echo "  $0 my_audio.wav --shots 10"
    echo ""
    echo "  # 自訂 shot 範圍 (使用 1-10 和 20-25)"
    echo "  $0 my_audio.wav --range '1-10,20-25'"
    echo ""
    echo "  # 使用特定的 shot (1, 5, 10, 15, 20)"
    echo "  $0 my_audio.wav --range '1,5,10,15,20'"
    echo ""
    echo "  # 不使用 10 字限制"
    echo "  $0 my_audio.wav --instruction none"
    echo ""
    echo "  # 完整範例"
    echo "  $0 examples/ldv/mix/boy1_papercup_LDV_041.wav \\"
    echo "     --output enhanced.wav \\"
    echo "     --shots 20 \\"
    echo "     --dataset optical_denoised"
    echo ""
}

# 解析命令列參數
INPUT_AUDIO=""
OUTPUT_AUDIO=""
MAX_TOKENS=8192

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output)
            OUTPUT_AUDIO="$2"
            shift 2
            ;;
        -s|--shots)
            SHOTS="$2"
            shift 2
            ;;
        -r|--range)
            SHOT_RANGE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -i|--instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --custom-inst)
            CUSTOM_INSTRUCTION="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}❌ 未知選項: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$INPUT_AUDIO" ]; then
                INPUT_AUDIO="$1"
            else
                echo -e "${RED}❌ 多餘的參數: $1${NC}"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 檢查必要參數
if [ -z "$INPUT_AUDIO" ]; then
    echo -e "${RED}❌ 錯誤: 缺少輸入音檔${NC}"
    echo ""
    show_usage
    exit 1
fi

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ 錯誤: 找不到 Docker${NC}"
    exit 1
fi

if ! docker images | grep -q "mimo-audio"; then
    echo -e "${RED}❌ 錯誤: 找不到 mimo-audio:latest Docker image${NC}"
    exit 1
fi

# 檢查輸入檔案
if [ ! -f "$INPUT_AUDIO" ]; then
    echo -e "${RED}❌ 錯誤: 找不到輸入檔案: $INPUT_AUDIO${NC}"
    exit 1
fi

# 設定輸出檔案
if [ -z "$OUTPUT_AUDIO" ]; then
    OUTPUT_AUDIO="outputs/enhanced_$(basename "$INPUT_AUDIO")"
fi

mkdir -p "$(dirname "$OUTPUT_AUDIO")"

# 轉換路徑（Docker 使用）
CONTAINER_INPUT="$INPUT_AUDIO"
CONTAINER_OUTPUT="$OUTPUT_AUDIO"

if [[ "$INPUT_AUDIO" == /* ]]; then
    REL_INPUT=$(realpath --relative-to="$(pwd)" "$INPUT_AUDIO")
    CONTAINER_INPUT="/app/$REL_INPUT"
fi

if [[ "$OUTPUT_AUDIO" == /* ]]; then
    REL_OUTPUT=$(realpath --relative-to="$(pwd)" "$OUTPUT_AUDIO")
    CONTAINER_OUTPUT="/app/$REL_OUTPUT"
else
    CONTAINER_OUTPUT="/app/$OUTPUT_AUDIO"
fi

# 顯示配置
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}📋 配置資訊${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓${NC} 輸入檔案: ${BLUE}$INPUT_AUDIO${NC}"
echo -e "${GREEN}✓${NC} 輸出檔案: ${BLUE}$OUTPUT_AUDIO${NC}"
echo -e "${GREEN}✓${NC} 資料集: ${BLUE}$DATASET${NC}"

if [ -n "$SHOT_RANGE" ]; then
    echo -e "${GREEN}✓${NC} Shot 範圍: ${BLUE}$SHOT_RANGE${NC} (自訂)"
else
    echo -e "${GREEN}✓${NC} Shot 數量: ${BLUE}$SHOTS${NC} (前 $SHOTS 個)"
fi

case $INSTRUCTION in
    "10-char")
        echo -e "${GREEN}✓${NC} Instruction: ${BLUE}10 字中文限制${NC}"
        ;;
    "none")
        echo -e "${GREEN}✓${NC} Instruction: ${BLUE}無限制${NC}"
        ;;
    "custom")
        echo -e "${GREEN}✓${NC} Instruction: ${BLUE}自訂${NC}"
        ;;
esac

echo -e "${GREEN}✓${NC} Max tokens: ${BLUE}$MAX_TOKENS${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 建立臨時 Python 腳本
TEMP_SCRIPT="./temp_inference_$(date +%s).py"

cat > "$TEMP_SCRIPT" << 'PYTHON_SCRIPT_START'
import sys
import os
from src.mimo_audio.mimo_audio import MimoAudio

# 從命令列參數取得配置
input_audio = sys.argv[1]
output_audio = sys.argv[2]
shots = int(sys.argv[3])
shot_range = sys.argv[4] if len(sys.argv) > 4 else ""
dataset = sys.argv[5] if len(sys.argv) > 5 else "optical_denoised"
instruction_type = sys.argv[6] if len(sys.argv) > 6 else "10-char"
custom_instruction = sys.argv[7] if len(sys.argv) > 7 else ""
max_tokens = int(sys.argv[8]) if len(sys.argv) > 8 else 8192

print("=" * 70)
print("🚀 載入模型...")
print("=" * 70)

# 載入模型
model = MimoAudio(
    model_path="models/MiMo-Audio-7B-Base",
    mimo_audio_tokenizer_path="models/MiMo-Audio-Tokenizer"
)

print("\n✅ 模型載入完成\n")

# 建立 instruction
if instruction_type == "10-char":
    instruction = "Enhance the audio quality and remove noise from the input speech. IMPORTANT: You MUST preserve the exact original speech content and transcript. The speech content is exactly 10 Chinese characters. Only improve the audio quality, do not change any words."
elif instruction_type == "none":
    instruction = "Enhance the audio quality and remove noise from the input speech. Preserve the original speech content."
elif instruction_type == "custom":
    instruction = custom_instruction
else:
    instruction = "Enhance the audio quality and remove noise from the input speech."

# 解析 shot 範圍
def parse_shot_range(range_str):
    """
    解析 shot 範圍字串
    例如: "1-10,15,20-25" -> [1,2,3,...,10,15,20,21,...,25]
    """
    if not range_str:
        return None
    
    indices = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 範圍 (例如 1-10)
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            # 單一數字
            indices.append(int(part))
    
    return sorted(set(indices))  # 去重並排序

# 建立 prompt examples
prompt_examples = []

# 根據資料集設定路徑
if dataset == "optical_denoised":
    mix_prefix = "examples/optical_denoised/mix/boy1_WOLDVlean_"
    spk_prefix = "examples/optical_denoised/spk1/boy1_papercup_clean_"
elif dataset == "optical":
    mix_prefix = "examples/optical/mix/boy1_papercup_"
    spk_prefix = "examples/optical/spk1/boy1_papercup_clean_"
elif dataset == "ldv":
    mix_prefix = "examples/ldv/mix/boy1_papercup_LDV_"
    spk_prefix = "examples/ldv/spk1/boy1_papercup_clean_"
else:
    raise ValueError(f"Unknown dataset: {dataset}")

# 決定使用哪些 shot
if shot_range:
    shot_indices = parse_shot_range(shot_range)
    print(f"使用自訂 shot 範圍: {shot_indices}")
    print(f"共 {len(shot_indices)} 個 shots")
else:
    shot_indices = list(range(1, shots + 1))
    print(f"使用前 {shots} 個 shots")

# 建立 examples
for i in shot_indices:
    prompt_examples.append({
        "input_audio": f"{mix_prefix}{i:03d}.wav",
        "output_audio": f"{spk_prefix}{i:03d}.wav",
        "output_transcription": "",
    })

print("=" * 70)
print("🎯 開始推論")
print("=" * 70)
print(f"配置: Exp(A++)")
print(f"資料集: {dataset}")
print(f"Shot count: {len(prompt_examples)}")
print(f"Instruction: {instruction_type}")
print(f"輸入: {input_audio}")
print(f"輸出: {output_audio}")
print("=" * 70 + "\n")

try:
    text_output = model.in_context_learning_s2s(
        instruction, 
        prompt_examples, 
        input_audio, 
        max_new_tokens=max_tokens, 
        output_audio_path=output_audio
    )
    
    print("\n" + "=" * 70)
    print("✅ 推論成功完成！")
    print("=" * 70)
    print(f"Text output: {text_output}")
    print(f"Enhanced audio saved to: {output_audio}")
    print("=" * 70)
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ 推論失敗！")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT_START

# 執行推論
echo -e "${YELLOW}⏳ 在 Docker 容器中執行推論...${NC}"
echo ""

# 準備參數
PYTHON_ARGS="$CONTAINER_INPUT $CONTAINER_OUTPUT $SHOTS"

if [ -n "$SHOT_RANGE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS \"$SHOT_RANGE\""
else
    PYTHON_ARGS="$PYTHON_ARGS \"\""
fi

PYTHON_ARGS="$PYTHON_ARGS \"$DATASET\" \"$INSTRUCTION\""

if [ "$INSTRUCTION" = "custom" ]; then
    PYTHON_ARGS="$PYTHON_ARGS \"$CUSTOM_INSTRUCTION\""
else
    PYTHON_ARGS="$PYTHON_ARGS \"\""
fi

PYTHON_ARGS="$PYTHON_ARGS \"$MAX_TOKENS\""

# 執行 Docker
docker run --gpus all --rm \
  -v "$(pwd)":/app \
  -w /app \
  mimo-audio:latest \
  bash -c "python $TEMP_SCRIPT $PYTHON_ARGS"

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
