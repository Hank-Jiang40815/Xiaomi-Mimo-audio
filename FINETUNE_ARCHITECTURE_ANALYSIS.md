# MiMo-Audio 微調架構分析

## 🔍 您的問題分析

### 問題 1: 微調會動到原先小米模型的 token 嗎？
**答案：不會！**

### 問題 2: 我們是透過調整 encoder 讓新的 token 能對應到原先小米的 token 嗎？
**答案：是的，但需要澄清一個重要概念！**

### 問題 3: 中間需要轉換器嗎？
**答案：不需要額外的轉換器！**

---

## 📐 MiMo-Audio 完整架構

基於代碼分析，MiMo-Audio 的架構如下：

```
音訊輸入 (Waveform)
    ↓
┌─────────────────────────────────────────────────────────────┐
│              MiMo-Audio-Tokenizer (1.2B 參數)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Mel-Spectrogram 轉換                              │   │
│  │     - 採樣率: 24kHz                                   │   │
│  │     - Hop length: 240                                 │   │
│  │     - N_mels: 80                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Audio Encoder (8 layers Transformer)             │   │
│  │     - Conv1d (n_mels → d_model=768)                  │   │
│  │     - Conv1d (stride=2, downsampling)                │   │
│  │     - 8 Transformer layers                            │   │
│  │     - RoPE position embedding                         │   │
│  │     ↓                                                  │   │
│  │  Hidden States (continuous representations)          │   │
│  │     - Shape: [T', 768]                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Residual Vector Quantizer (RVQ)                  │   │
│  │     - 8-layer RVQ stack                               │   │
│  │     - Codebook size: 1024 per layer                  │   │
│  │     - 25 Hz token rate                                │   │
│  │     ↓                                                  │   │
│  │  Discrete Codes (tokens)                             │   │
│  │     - Shape: [8, T]  (8 codebooks)                   │   │
│  │     - Values: 0-1023 (discrete token IDs)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Audio Decoder (8 layers Transformer)             │   │
│  │     - Dequantize codes → Hidden states              │   │
│  │     - 8 Transformer layers                            │   │
│  │     - ConvTranspose (upsampling)                     │   │
│  │     - Vocoder (30 layers Transformer)                │   │
│  │     ↓                                                  │   │
│  │  Reconstructed Audio (Waveform)                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
            Discrete Audio Tokens (8 × T)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│           MiMo-Audio-7B (語言模型)                            │
│  - 基於 Qwen2-7B                                              │
│  - 處理離散的 audio tokens                                    │
│  - 生成新的 audio tokens                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 關鍵理解：Encoder 的角色

### Encoder 做什麼？

```python
# 從代碼可以看到：
class AudioEncoder(nn.Module):
    def encode(self, input_features, ...):
        # 1. 提取連續特徵
        hidden_states = self.get_features(input_features)
        
        # 2. 量化為離散 tokens
        codes = self.quantizer.encode(hidden_states)  # ← 關鍵！
        
        # 3. 可以反量化回連續表示
        hidden_states = self.quantizer.decode(codes)
        
        return hidden_states, codes
```

### Quantizer (RVQ) 的作用

```python
class ResidualVectorQuantizer:
    def encode(self, x):
        """
        輸入: 連續的 hidden states [T', 768]
        輸出: 離散的 codes [8, T]  (每個值 0-1023)
        """
        # 將連續向量映射到最近的 codebook entry
        codes = find_nearest_codebook_entry(x)
        return codes
    
    def decode(self, codes):
        """
        輸入: 離散的 codes [8, T]
        輸出: 連續的 hidden states [T', 768]
        """
        # 從 codebook 查詢對應的向量
        hidden_states = lookup_codebook(codes)
        return hidden_states
```

---

## ✅ 微調策略的正確性分析

### 您的理解是正確的！

```
噪聲音訊 → [Encoder] → Hidden States → [Quantizer] → Discrete Codes
                ↑
              微調這裡
```

### 為什麼這樣做有效？

1. **Quantizer (Codebook) 是固定的**
   - Codebook 包含 1024 個預定義的向量
   - 這些向量是預訓練時學到的
   - **不會被微調改變**

2. **Encoder 學習更好的映射**
   - 原本：`噪聲音訊 → 差的 hidden states → 不好的 codes`
   - 微調後：`噪聲音訊 → 好的 hidden states → 正確的 codes`

3. **不需要轉換器**
   - Encoder 的輸出會自動通過 Quantizer
   - Quantizer 將連續向量映射到最近的 codebook entry
   - **只要 Encoder 輸出更接近"正確的" hidden states，Quantizer 就會選擇正確的 codes**

---

## 🔬 具體例子

### 場景：處理嚴重噪聲音訊

**原始模型：**
```
噪聲音訊 "你好" (很多噪聲)
    ↓
[Encoder] → 提取特徵（受噪聲影響，不準確）
    ↓
Hidden States: [混亂的向量]
    ↓
[Quantizer] → 找最近的 codebook entry
    ↓
Codes: [234, 567, 123, ...]  ← 錯誤的 codes
    ↓
[Decoder] → 重建
    ↓
輸出音訊: "哩窩" (錯誤)
```

**微調後的模型：**
```
噪聲音訊 "你好" (很多噪聲)
    ↓
[Fine-tuned Encoder] → 學會從噪聲中提取正確特徵
    ↓
Hidden States: [清晰的向量] ← 更接近乾淨音訊的表示
    ↓
[Quantizer] → 找最近的 codebook entry (同一個 codebook!)
    ↓
Codes: [789, 234, 456, ...]  ← 正確的 codes
    ↓
[Decoder] → 重建
    ↓
輸出音訊: "你好" (正確)
```

---

## 💡 訓練目標

### 理想的損失函數

```python
def training_objective(noisy_audio, clean_audio):
    # 1. 編碼噪聲音訊
    noisy_hidden = encoder(noisy_audio)
    
    # 2. 編碼乾淨音訊（目標）
    clean_hidden = encoder_pretrained(clean_audio)  # 凍結
    
    # 3. 讓噪聲音訊的 hidden states 接近乾淨音訊的
    loss = MSE(noisy_hidden, clean_hidden)
    
    return loss
```

### 為什麼這樣有效？

- **Codebook 是固定的**：預訓練時學到的"正確"音訊表示
- **Encoder 學習映射**：從噪聲 → 正確的連續表示
- **Quantizer 自動處理**：將連續表示映射到離散 codes
- **不需要改變 token 空間**：token 仍然是 0-1023

---

## 🚨 重要澄清

### ❌ 錯誤理解
"微調會創建新的 token，需要轉換器將新 token 轉換為舊 token"

### ✅ 正確理解
"微調讓 Encoder 學會從噪聲音訊中提取更好的特徵，這些特徵會被 Quantizer 自動映射到**相同的** codebook（相同的 token 空間）"

### 關鍵點

1. **Token 空間不變**
   - 預訓練：tokens ∈ {0, 1, ..., 1023} × 8 layers
   - 微調後：tokens ∈ {0, 1, ..., 1023} × 8 layers
   - **完全相同！**

2. **Codebook 不變**
   - 預訓練時學到的 codebook 是固定的
   - 微調時 **凍結 Quantizer**
   - 只訓練 Encoder

3. **改變的是映射**
   - 原本：`噪聲音訊 → [Encoder] → 不好的 hidden → 錯誤的 codes`
   - 微調：`噪聲音訊 → [Better Encoder] → 好的 hidden → 正確的 codes`

---

## 📝 微調實現要點

### 正確的微調方式

```python
class FineTuneEncoder:
    def __init__(self, tokenizer_model):
        self.encoder = tokenizer_model.encoder  # 微調這個
        self.quantizer = tokenizer_model.encoder.quantizer  # 凍結這個
        
        # 凍結 quantizer
        for param in self.quantizer.parameters():
            param.requires_grad = False
    
    def forward(self, noisy_audio, clean_audio):
        # 編碼噪聲音訊
        noisy_features = self.encoder.get_features(noisy_audio)
        
        # 編碼乾淨音訊（作為目標）
        with torch.no_grad():
            clean_features = self.encoder.get_features(clean_audio)
        
        # 損失：讓噪聲特徵接近乾淨特徵
        loss = F.mse_loss(noisy_features, clean_features)
        
        return loss
```

### 為什麼不訓練 Quantizer？

1. **Quantizer 是字典**：它定義了 token 的語義
2. **改變它會破壞兼容性**：LLM 期望特定的 token 分佈
3. **Encoder 更靈活**：學習更好的映射就夠了

---

## 🎓 總結

### 您的理解基本正確！

✅ **微調不會動到原先的 token**
- Token 空間（0-1023）保持不變
- Codebook 保持不變

✅ **通過調整 Encoder 讓新的表示對應到正確的 token**
- Encoder 學習從噪聲中提取更好的特徵
- Quantizer 自動將這些特徵映射到正確的 codes

✅ **不需要額外的轉換器**
- Quantizer 本身就是"轉換器"
- 它將連續的 hidden states 轉換為離散的 codes
- 微調只是讓 Encoder 輸出更好的 hidden states

### 完整流程

```
微調前：
噪聲 → [Encoder] → 差的 hidden → [Quantizer] → 錯誤的 codes → [LLM] → 差的輸出

微調後：
噪聲 → [Better Encoder] → 好的 hidden → [Quantizer] → 正確的 codes → [LLM] → 好的輸出
         ↑                                   ↑
      只訓練這個                         固定不變的 codebook
```

---

## 🔧 實際微調建議

### 1. 只微調 Encoder 的特定層
```python
# 例如：只微調最後 2-3 層
for name, param in encoder.named_parameters():
    if 'layer_6' in name or 'layer_7' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### 2. 使用 LoRA 減少參數量
```python
# 只在關鍵層添加 LoRA
apply_lora_to_layers(encoder, layers=[6, 7], rank=8)
```

### 3. 確保 Quantizer 完全凍結
```python
for param in tokenizer.encoder.quantizer.parameters():
    param.requires_grad = False
    
# 驗證
assert all(not p.requires_grad for p in tokenizer.encoder.quantizer.parameters())
```

### 4. 驗證 token 空間一致性
```python
# 微調前後，相同的乾淨音訊應該產生相同的 codes
clean_audio = load_audio("clean.wav")
codes_before = pretrained_tokenizer.encode(clean_audio)
codes_after = finetuned_tokenizer.encode(clean_audio)
assert torch.allclose(codes_before, codes_after, atol=1e-5)
```

---

## ⚠️ 注意事項

1. **不要動 Decoder**
   - Decoder 是從 codes 重建音訊的
   - 微調 Decoder 沒有意義，因為 codes 還是會經過 LLM

2. **不要動 Quantizer**
   - Quantizer 的 codebook 定義了 token 語義
   - 改變它會破壞與 LLM 的兼容性

3. **只微調 Encoder**
   - 讓它學會從噪聲中提取更魯棒的特徵
   - 這些特徵會被 Quantizer 映射到正確的 codes

4. **驗證兼容性**
   - 微調後的模型應該與原始 LLM 完全兼容
   - Token 空間不應改變

---

**結論：您的理解是正確的！微調 Encoder 不會改變 token 空間，只是讓 Encoder 學會更好的映射。不需要額外的轉換器，因為 Quantizer 本身就負責將連續表示轉換為離散 codes。**
