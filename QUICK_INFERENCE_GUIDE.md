# Quick Inference - Exp(A+) 配置

快速使用 exp(A+) 實驗配置（commit f56da64）來推論新的音檔

## 🎯 配置說明

**實驗**: exp(A+) - 10 字中文限制  
**Commit**: f56da64568dc62506041382b32ff92c25f4d1901  
**模型**: MiMo-Audio-7B-Base  
**方法**: 40-shot In-Context Learning

**Instruction**:
```
Enhance the audio quality and remove noise from the input speech. 
IMPORTANT: You MUST preserve the exact original speech content and transcript. 
The speech content is exactly 10 Chinese characters. 
Only improve the audio quality, do not change any words.
```

**訓練範例**: optical_denoised dataset (001-040, 共 40 組)

## 🚀 使用方式

### 基本用法

```bash
./quick_inference_expA+.sh <input_audio.wav>
```

輸出會自動儲存到 `outputs/enhanced_<檔名>.wav`

### 指定輸出路徑

```bash
./quick_inference_expA+.sh <input_audio.wav> <output_audio.wav>
```

## 📝 實際範例

### 範例 1: 基本使用

```bash
# 推論單個檔案
./quick_inference_expA+.sh my_noisy_audio.wav

# 輸出: outputs/enhanced_my_noisy_audio.wav
```

### 範例 2: 指定輸出路徑

```bash
./quick_inference_expA+.sh \
    examples/ldv/mix/boy1_papercup_LDV_041.wav \
    results/ldv_041_enhanced.wav
```

### 範例 3: 批次處理

```bash
# 處理多個檔案
for file in my_audios/*.wav; do
    echo "處理: $file"
    ./quick_inference_expA+.sh "$file"
done

# 所有輸出會在 outputs/ 目錄
```

### 範例 4: 使用 LDV 資料集

```bash
# 測試 LDV 檔案
./quick_inference_expA+.sh \
    examples/ldv/mix/boy1_papercup_LDV_041.wav \
    outputs/ldv_041_expA+.wav
```

## 📊 預期輸出

執行後會看到：

```
╔═══════════════════════════════════════════════════════════════╗
║     Exp(A+) 快速推論工具 - 10字限制配置                       ║
╚═══════════════════════════════════════════════════════════════╝

✓ 輸入檔案: my_noisy_audio.wav
✓ 輸出檔案: outputs/enhanced_my_noisy_audio.wav

============================================================
🚀 載入模型...
============================================================

✅ 模型載入完成

============================================================
🎯 開始推論
============================================================
配置: Exp(A+) - 10 字中文限制
Shot count: 40
輸入: my_noisy_audio.wav
輸出: outputs/enhanced_my_noisy_audio.wav
============================================================

⏳ 執行推論中...

============================================================
✅ 推論成功完成！
============================================================
Text output: <轉錄文字>
Enhanced audio saved to: outputs/enhanced_my_noisy_audio.wav
============================================================

╔═══════════════════════════════════════════════════════════════╗
║     推論完成                                                   ║
╚═══════════════════════════════════════════════════════════════╝

✅ 成功生成增強音檔
   檔案: outputs/enhanced_my_noisy_audio.wav
   大小: 143K

💡 播放指令:
   ffplay -autoexit -nodisp outputs/enhanced_my_noisy_audio.wav
```

## ⚙️ 技術細節

### 模型配置
- **Model**: models/MiMo-Audio-7B-Base
- **Tokenizer**: models/MiMo-Audio-Tokenizer
- **Device**: CUDA (GPU)
- **Max tokens**: 8192

### ICL 範例
- **數量**: 40 組 (001-040)
- **來源**: examples/optical_denoised/
- **格式**: noisy (mix) → clean (spk1)

### Instruction 重點
1. **內容保留**: 必須保持原始語音內容和轉錄文字
2. **字數限制**: 明確說明為 10 個中文字
3. **品質提升**: 只改善音訊品質，不改變任何文字

## 🔧 故障排除

### 問題 1: 找不到輸入檔案

```
❌ 錯誤: 找不到輸入檔案: xxx.wav
```

**解決**: 確認檔案路徑正確，使用絕對路徑或相對路徑

```bash
# 使用絕對路徑
./quick_inference_expA+.sh /full/path/to/audio.wav

# 使用相對路徑
./quick_inference_expA+.sh ./examples/ldv/mix/boy1_papercup_LDV_041.wav
```

### 問題 2: 找不到模型檔案

**解決**: 確認模型已下載到正確位置

```bash
ls models/MiMo-Audio-7B-Base/
ls models/MiMo-Audio-Tokenizer/
```

### 問題 3: 找不到訓練範例

**解決**: 確認 optical_denoised 資料集存在

```bash
ls examples/optical_denoised/mix/boy1_WOLDVlean_001.wav
ls examples/optical_denoised/spk1/boy1_papercup_clean_001.wav
```

### 問題 4: CUDA out of memory

**解決**: 
1. 確認沒有其他程式佔用 GPU
2. 考慮使用更少的 shot (修改腳本中的 range(1, 41))

```bash
# 檢查 GPU 使用
nvidia-smi
```

## 📈 效能參考

基於原始實驗記錄：

- **模型載入**: ~2-3 秒
- **單檔推論**: ~15-20 秒 (40-shot ICL)
- **記憶體使用**: ~10-15GB GPU RAM
- **輸出大小**: 約 120-150KB (依音訊長度)

## 🔄 與其他工具比較

| 工具 | 配置 | Shot 數 | 速度 | 適用 |
|------|------|---------|------|------|
| `quick_inference_expA+.sh` | Exp(A+) 固定 | 40 | 中 | 快速測試 exp(A+) |
| `run_inference.py` | 可選多種 | 10-40 | 快-中 | 彈性配置 |
| `experiment_audio_enhancement.py` | 完整實驗 | 40 | 慢 | 研究用途 |

## 💡 使用建議

### ✅ 適合使用的情況

- 快速測試 exp(A+) 配置
- 你的音訊是 **10 個中文字** 左右
- 需要簡單的指令快速推論
- 想複現 commit f56da64 的配置

### ❌ 不適合的情況

- 需要不同的 instruction → 用 `run_inference.py`
- 需要不同的 shot 數 → 用 `run_inference.py`
- 需要實驗記錄 → 用 `experiment_audio_enhancement.py`
- 音訊長度遠超 10 字 → 考慮其他配置

## 📚 相關文件

- **實驗記錄**: 查看 commit f56da64 的訊息
- **完整配置**: `experiment_audio_enhancement.py`
- **彈性推論**: `run_inference.py` + `INFERENCE_GUIDE.md`
- **其他實驗**: `EXPERIMENT_LOG.md`

## 🎓 原始實驗結果

根據 commit f56da64 的實驗：

**字數限制效果**:
- 041: 生成 11 字（超出）
- 042: 生成 10 字（符合）✅
- 043: 生成 8 字（不足）

**結論**: 字數限制約束的遵守率為 33.3%，模型優先考慮語法自然度

## 🔗 相關 Commit

- **當前配置**: f56da64 (exp A+ - 10 字限制)
- **前一版本**: 9193483 (exp A - 內容保留)
- **基礎版本**: 1b13e11 (40-shot ICL)

---

**快速開始**: `./quick_inference_expA+.sh your_audio.wav` 🚀
