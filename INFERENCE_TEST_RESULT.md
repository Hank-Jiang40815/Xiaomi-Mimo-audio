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

### 根本原因（2025-11-14 更新：此分析有誤）

⚠️ **重要更正**: 經過深入檢查源碼，發現 `input_lens` 的使用方式是**正確的**！

```python
# ✅ 正確的 input_lens 計算
input_lens = torch.tensor([mel_spec.shape[2]], device=device)
```

**證據**：
1. `MiMoAudioTokenizer.encode()` 接受的 `input_lens` 就是 mel 的時間維度
2. 源碼 `src/mimo_audio/mimo_audio.py` 第 257 行也是用 `mel.size(0)` 作為 input_lens
3. 訓練代碼的 input_lens 計算是正確的

### 真正的問題
實際問題出在 `test_finetuned_inference.py` 的 API 使用錯誤：
1. `model.encode()` 返回 4 個值，但只接收了 1 個
2. `model.decode()` 需要 codes（離散tokens），但傳入了錯誤的參數

## 結論

### 當前狀態
**微調後的 Encoder 無法直接用於 Inference**，原因：
1. 訓練時 `input_lens` 計算有誤
2. Encoder 的 forward pass 在 inference 時失敗

### 建議行動

#### 優先級 P0 (立即修復) - ✅ 已修正
- [x] **修正 test_finetuned_inference.py 的 API 調用**
  ```python
  # ✅ 正確的方式
  hidden_states, hidden_states_packed, output_length, codes = model.encode(
      mels=mel_spec,
      input_lens=input_lens,
      use_quantizer=True
  )
  reconstructed_audio = model.decode(codes)
  ```

- [ ] **測試修正後的 Inference**
  - 使用現有的 checkpoint 測試
  - 驗證音質改善效果
  - 確認端到端 pipeline 正常運作

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

### 正確的 input_lens 處理（更正版）
```python
# ✅ finetune_encoder.py 的當前實現是正確的

# 轉換為 Mel Spectrogram
mel_spec = mel_transform(waveform)  # [1, 128, T]

# 使用 Mel 的時間維度作為 input_lens
mel_len = mel_spec.shape[2]
input_lens = torch.tensor([mel_len], device=device)

# 傳入 encoder
features = encoder.get_features(
    input_features=mel_spec,
    output_length=encoder.get_output_length(input_lens)
)
```

### MelSpectrogram 與 input_lens 的關係
```
✅ 正確理解：
- input_lens = mel spectrogram 的時間維度
- 不是原始音訊的樣本數！

例如:
- 音訊: 62399 samples (2.6 秒 @ 24kHz)
- Mel: 260 frames (hop_length=240)
- input_lens = 260 ✅
```

## 實驗記錄影響（更正）

✅ **FINETUNE_EXPERIMENT_R32_E100 實驗仍然有效**
- 訓練過程的 `input_lens` 計算是正確的
- Loss 下降是真實的特徵匹配改善
- 需要測試 inference 來驗證實際降噪效果

## 下一步（更新）

1. ✅ **已修正**: test_finetuned_inference.py 的 API 調用
2. **測試**: 使用修正後的 inference 腳本測試現有 checkpoint
3. **驗證**: 確認模型可以正常 encode-decode
4. **評估**: 用客觀指標 (SI-SDR, PESQ) 和主觀聽感評估降噪效果
5. **整合**: 將成功的 encoder 整合到完整的 MiMo-Audio pipeline

---

**報告建立時間**: 2025-11-14 04:10  
**更新時間**: 2025-11-14 06:30  
**發現者**: Hank + GitHub Copilot  
**嚴重程度**: 🟡 Medium (只影響 inference 測試腳本，訓練代碼正確)  
**狀態**: ✅ 已修正 inference 腳本
