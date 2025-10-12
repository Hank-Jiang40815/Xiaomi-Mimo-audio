# 音頻增強實驗分析：使用 In-Context Learning

## 🎯 實驗目標

將 **低質量音頻（LDV - Low-quality/Distorted Version）** 轉換為 **高質量乾淨音頻（clean）**

## 📊 任務對比分析

### 原始範例 vs 新任務

| 維度 | 原始音色轉換 | 音頻增強（新任務）|
|------|-------------|------------------|
| **輸入** | 說話人 A 的聲音 | 低質量/有噪音音頻 |
| **輸出** | 說話人 B 的聲音 | 高質量/乾淨音頻 |
| **保持不變** | 文字內容 | 文字內容 + 說話人特徵 |
| **改變內容** | 音色特徵（pitch, timbre）| 音質（noise, clarity, fidelity）|
| **任務類型** | Voice Conversion | Audio Enhancement/Denoising |
| **應用場景** | 語音克隆、配音 | 語音修復、通話增強 |

## 🔬 可行性分析

### ✅ 支持可行性的理由

#### 1. **相同的模型架構和能力**
```
In-Context Learning 的核心機制：
  示例 1: [輸入音頻 A] → [輸出音頻 A']
  示例 2: [輸入音頻 B] → [輸出音頻 B']
  ...
  示例 N: [輸入音頻 N] → [輸出音頻 N']
  ────────────────────────────────────
  測試:   [目標音頻 X] → [模型生成 X']
  
模型學習的是「轉換模式」，不限定於特定轉換類型
```

#### 2. **任務本質相似**
- **音色轉換**: 移除說話人 A 的特徵 + 添加說話人 B 的特徵
- **音頻增強**: 移除噪音/失真特徵 + 保持/增強語音特徵

兩者都是「**特徵層面的轉換**」任務

#### 3. **模型已證明的能力**
- MiMo-Audio Base 可以處理複雜的音頻特徵變換
- 已成功進行音色轉換（更困難的任務）
- 音頻增強可視為「特殊的音色轉換」：轉換到「無噪音版本的自己」

#### 4. **文獻支持**
類似的多模態大模型（如 AudioLM, MusicGen）已展示：
- Few-shot 音頻任務學習能力
- 跨任務遷移能力
- 音頻修復和增強能力

### ⚠️ 潛在限制和挑戰

#### 1. **示例數量不足**
- **當前可用**: 僅 2 對 LDV→clean 示例
- **原始範例**: 使用 5 對示例
- **影響**: 示例越少，模型學習的模式越不穩定
- **建議**: 
  - 最少 3-5 對示例
  - 示例應涵蓋不同類型的噪音/失真

#### 2. **音頻配對要求**
**關鍵要求**: LDV 和 clean 版本必須是**完全相同的語音內容**

```
✅ 正確配對:
  LDV_001: "今天天氣很好"（有噪音）
  clean_001: "今天天氣很好"（無噪音）
  
❌ 錯誤配對:
  LDV_001: "今天天氣很好"（有噪音）
  clean_001: "明天會下雨"（無噪音）
```

驗證方法：
- 時長應該相近（✅ 我們的數據：001 都是 2.66s，002 都是 2.82s）
- 文字內容相同（需要人工確認或 ASR 驗證）

#### 3. **採樣率差異**
```
LDV files:   16000 Hz
clean files: 22050 Hz
```

**可能的影響**:
- 模型可能學習到「提升採樣率」的模式
- 也可能將這視為「質量提升」的一部分
- **建議**: 如果可能，統一採樣率到 22050 Hz

#### 4. **指令（Instruction）的重要性**

指令質量直接影響模型理解：

**推薦指令選項**:
```python
# 選項 1: 直接明確
"Enhance the audio quality and remove noise from the input speech."

# 選項 2: 更具體
"Remove background noise and distortion to produce clean, high-quality speech."

# 選項 3: 類比原始任務
"Convert low-quality noisy audio to high-quality clean audio."

# 選項 4: 技術導向
"Denoise and enhance the speech signal while preserving the speaker identity and speech content."
```

## 🧪 實驗設計

### 實驗方案 A: 使用單個示例（最小配置）

```python
instruction = "Enhance the audio quality and remove noise from the input speech."

input_audio = "examples/boy1_papercup_LDV_001.wav"

prompt_examples = [
    {
        "input_audio": "examples/boy1_papercup_LDV_002.wav",
        "output_audio": "examples/boy1_papercup_clean_002.wav",
        "output_transcription": "Transcription of 002",
    },
]
```

**預期**: 可能有效，但效果不穩定

### 實驗方案 B: 交叉驗證（推薦）

```python
# Test 1: 使用 002 作為示例，增強 001
prompt_examples = [002]
input = 001
output = enhancement_001_from_002.wav

# Test 2: 使用 001 作為示例，增強 002
prompt_examples = [001]
input = 002
output = enhancement_002_from_001.wav
```

**評估**: 比較兩個結果與對應的 clean 版本

### 實驗方案 C: 增加 ESD 示例（如果任務相似）

如果 ESD 數據集也有 noisy→clean 的配對，可以混合使用：

```python
prompt_examples = [
    # boy1_papercup 示例
    {"input": "boy1_LDV_002", "output": "boy1_clean_002", ...},
    # ESD 示例（如果有類似的噪音增強配對）
    {"input": "ESD_noisy_A", "output": "ESD_clean_A", ...},
    {"input": "ESD_noisy_B", "output": "ESD_clean_B", ...},
]
```

## 📈 評估方法

### 1. **主觀評估（聽感）**
```bash
# 播放三個版本進行對比
play examples/boy1_papercup_LDV_001.wav        # 原始低質量
play examples/audio_enhancement_result.wav     # 模型增強結果
play examples/boy1_papercup_clean_001.wav      # 參考高質量
```

評估維度：
- [ ] 噪音移除程度
- [ ] 語音清晰度
- [ ] 說話人特徵保持
- [ ] 自然度（無過度處理痕跡）

### 2. **客觀指標**

#### 音頻質量指標
```python
from pesq import pesq
from pystoi import stoi
import librosa
import numpy as np

# PESQ (Perceptual Evaluation of Speech Quality): 1.0-4.5, 越高越好
pesq_score = pesq(22050, reference, enhanced, 'wb')

# STOI (Short-Time Objective Intelligibility): 0-1, 越高越好
stoi_score = stoi(reference, enhanced, 22050, extended=False)

# SNR (Signal-to-Noise Ratio)
def calculate_snr(clean, noisy):
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((noisy - clean) ** 2)
    return 10 * np.log10(signal_power / noise_power)
```

#### 相似度指標
```python
# 與參考 clean 版本的相似度
from scipy.spatial.distance import cosine
from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()
clean_embed = encoder.embed_utterance(clean_wav)
enhanced_embed = encoder.embed_utterance(enhanced_wav)
similarity = 1 - cosine(clean_embed, enhanced_embed)
```

## 🎯 預期結果與風險

### 樂觀情況 ✅
- 模型成功學習 LDV→clean 的轉換模式
- 生成的音頻質量接近 clean 版本
- 保持說話人特徵和語音內容

**成功指標**:
- PESQ > 3.0
- STOI > 0.85
- 主觀聽感明顯改善

### 中性情況 ⚠️
- 部分改善但未完全達到 clean 質量
- 可能引入輕微失真
- 需要調整指令或增加示例

### 悲觀情況 ❌
- 模型未能理解增強任務
- 輸出與輸入差異不大，或變得更差
- 可能的原因：
  - 示例數量太少
  - 指令不清晰
  - 任務超出模型訓練範圍

## 💡 改進策略

### 如果初步實驗失敗

#### 策略 1: 豐富示例
```python
# 尋找更多配對數據
# 或者使用數據增強創建更多示例
- 不同噪音類型（白噪音、粉紅噪音、環境噪音）
- 不同 SNR 等級
- 不同失真類型
```

#### 策略 2: 優化指令
```python
# 嘗試不同的指令表述
instructions = [
    "Enhance the audio quality and remove noise.",
    "Remove background noise while keeping the speech clear.",
    "Convert noisy speech to clean high-quality speech.",
    "Denoise and improve the speech signal quality.",
]
```

#### 策略 3: 混合任務示例
```python
# 結合音色轉換和音頻增強
# 讓模型理解「保持內容，改變特徵」的通用模式
prompt_examples = [
    # 音色轉換示例
    {"input": "speaker_A", "output": "speaker_B", ...},
    # 音頻增強示例  
    {"input": "noisy", "output": "clean", ...},
]
```

#### 策略 4: 嘗試 SFT 模型（備選方案）
```python
# 如果 Base 模型效果不佳，嘗試用 SFT 模型的指令模式
model = MimoAudio("models/MiMo-Audio-7B-Instruct", ...)

# 直接指令式調用（如果 SFT 模型支持）
result = model.audio_enhancement_sft(
    input_audio,
    instruct="Remove noise and enhance audio quality"
)
```

## 📝 實驗執行

### 使用 Docker 運行實驗

```bash
# 運行音頻增強實驗
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/examples:/app/examples \
  mimo-audio:latest \
  python3.12 experiment_audio_enhancement.py
```

### 預期執行時間
- 模型載入: ~8-10 秒
- 推理時間: ~10-30 秒（取決於音頻長度和示例數量）
- 總時間: ~30-60 秒

## 🎓 理論基礎

### In-Context Learning 的工作原理

1. **Pattern Recognition**
   ```
   模型觀察示例對，識別轉換模式：
   - 輸入的共同特徵（低質量、有噪音）
   - 輸出的共同特徵（高質量、乾淨）
   - 轉換規則（去除什麼、保持什麼）
   ```

2. **Few-Shot Learning**
   ```
   少量示例 → 泛化能力
   - 不需要大量數據
   - 不需要微調模型
   - 依賴預訓練的語音理解能力
   ```

3. **Transfer Learning**
   ```
   預訓練知識 + 任務示例 = 新任務能力
   - 模型已知：語音結構、音素、韻律
   - 從示例學習：這個特定的轉換任務
   ```

### 為什麼可能成功

1. **音頻增強本質上是特徵過濾**
   - 保留：語音內容、說話人特徵、韻律
   - 移除：噪音、失真、混響

2. **模型有相關能力**
   - 已證明可以進行音色轉換（更複雜）
   - 理解音頻的多層次表示
   - 可以學習複雜的映射關係

3. **任務對齊性**
   - Speech-to-Speech 任務框架相同
   - 都需要保持語音內容
   - 都是特徵層面的轉換

## 📚 相關研究

類似方法在其他模型中的成功案例：
- **AudioLM**: Few-shot 音頻生成
- **Vall-E**: Zero-shot TTS with in-context learning
- **AudioGPT**: 多任務音頻處理
- **Speech Enhancement Transformers**: 基於 attention 的去噪

## 🚀 下一步計劃

1. **立即執行**: 運行 `experiment_audio_enhancement.py`
2. **評估結果**: 使用主觀和客觀指標
3. **迭代優化**: 根據結果調整策略
4. **文檔記錄**: 記錄成功/失敗的配置和結果

---

## 🎯 結論

**理論上完全可行！** 🟢

主要依據：
1. ✅ 相同的模型架構和 few-shot learning 機制
2. ✅ 任務本質相似（都是 speech-to-speech 特徵轉換）
3. ✅ 模型已展示類似的轉換能力
4. ✅ 有配對的訓練示例數據

**成功關鍵要素**:
- 📌 確保 LDV 和 clean 是完全配對的
- 📌 提供清晰的任務指令
- 📌 至少 2-3 對高質量示例
- 📌 評估和迭代優化

**建議**: 立即嘗試！即使只有 1-2 個示例，也值得測試。模型的 few-shot 能力可能超出預期。
