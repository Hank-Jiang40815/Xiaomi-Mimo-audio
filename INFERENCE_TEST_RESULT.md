# Inference 測試結果報告

## 測試日期
2025-11-14

## 測試目的
驗證微調後的 Encoder 是否可以正常進行 Inference

## 測試配置
- Checkpoint: `outputs/optical_lora_r32_e100/checkpoint_epoch_2.pt`
- 測試音訊: `examples/optical/mix/boy1_WOLDV_050.wav`
- 模型: LoRA Rank-32, Alpha-64

## 測試結果

### ✅ 成功部分
1. **模型載入**: ✅ 成功
   - 基礎 tokenizer 正常載入
   - LoRA 權重正確注入 (192 層)
   - Checkpoint 正常載入 (4 個參數)

2. **音訊預處理**: ✅ 成功
   - 音訊載入正常 (2.60 秒, 24kHz)
   - Mel Spectrogram 轉換正常 (Shape: [1, 128, 260])

### ❌ 失敗部分
3. **Encoder Inference**: ❌ 失敗
   - 錯誤類型: `RuntimeError: CUDA error: device-side assert triggered`
   - 錯誤位置: `unpack_hidden_states()` 函數
   - 根本原因: `input_lens` 計算錯誤

## 問題分析

### 根本原因
在 `finetune_encoder.py` 訓練腳本中：

```python
# ❌ 錯誤的 input_lens 計算
input_lens = torch.tensor([mel_spec.shape[2]], device=device)
```

這個值是 **mel spectrogram 的時間維度**，但 tokenizer.encode() 期望的是 **原始音訊的樣本數**。

### 證據
1. 訓練時使用: `input_lens = torch.tensor([mel_spec.shape[2]])`
2. Tokenizer 內部會根據 input_lens 進行 padding/unpacking
3. 當 input_lens 不匹配時，會觸發 index out of bounds error

### 影響範圍
- ❌ 訓練過程中的 `input_lens` 計算可能不正確
- ❌ 這可能影響了訓練效果
- ❌ Inference 無法正常運行

## 結論

### 當前狀態
**微調後的 Encoder 無法直接用於 Inference**，原因：
1. 訓練時 `input_lens` 計算有誤
2. Encoder 的 forward pass 在 inference 時失敗

### 建議行動

#### 優先級 P0 (立即修復)
- [ ] **修正 finetune_encoder.py 中的 input_lens 計算**
  ```python
  # ✅ 正確的方式
  # input_lens 應該是原始音訊的長度（樣本數）
  input_lens = torch.tensor([waveform.shape[1]], device=device)
  ```

- [ ] **重新訓練模型**
  - 使用修正後的 finetune_encoder.py
  - 可以使用較少的 epochs (20-30) 快速驗證
  - 同時修復資料洩漏問題

#### 優先級 P1 (後續驗證)
- [ ] **重新測試 Inference**
  - 使用新訓練的 checkpoint
  - 驗證 encoding 過程
  - 測試完整的 encode-decode pipeline

#### 優先級 P2 (長期改進)
- [ ] **整合到完整 pipeline**
  - 將微調後的 encoder 整合到 MiMo-Audio 主模型
  - 實現端到端的音訊增強

## 技術細節

### 正確的 input_lens 處理
```python
# finetune_encoder.py 應該這樣修改:

# 載入音訊 (waveform)
waveform, sr = torchaudio.load(audio_path)

# 記錄原始長度
original_length = waveform.shape[1]

# 轉換為 Mel Spectrogram
mel_spec = mel_transform(waveform)

# 使用原始音訊長度
input_lens = torch.tensor([original_length], device=device)

# 傳入 tokenizer
encoded = tokenizer.encode(mel_spec, input_lens=input_lens)
```

### MelSpectrogram 與 音訊長度的關係
```
原始音訊長度 (samples) = sr * duration
Mel 時間維度 = (原始音訊長度 - n_fft) / hop_length + 1

例如:
- 音訊: 62399 samples (2.6 秒 @ 24kHz)
- Mel: 260 frames (hop_length=240)
- 關係: (62399 - 1024) / 240 + 1 ≈ 260
```

## 實驗記錄影響

由於發現了這個重大問題，之前的訓練實驗結果需要標註：

⚠️ **FINETUNE_EXPERIMENT_R32_E100 實驗有效性存疑**
- 訓練過程使用了錯誤的 `input_lens` 計算
- Loss 下降可能不代表真實的降噪能力
- 需要修正後重新訓練以獲得可靠結果

## 下一步

1. **立即**: Commit 這份分析報告
2. **修正**: finetune_encoder.py 的 input_lens 計算
3. **重訓**: 使用修正後的程式碼重新訓練 (建議先用 20 epochs 快速驗證)
4. **驗證**: 確認新模型可以正常 inference
5. **評估**: 用客觀指標 (SI-SDR, PESQ) 評估真實降噪效果

---

**報告建立時間**: 2025-11-14 04:10  
**發現者**: Hank + GitHub Copilot  
**嚴重程度**: 🔴 Critical (影響所有 inference 和訓練有效性)
