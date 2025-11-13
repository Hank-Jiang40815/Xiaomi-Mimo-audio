# Optical Dataset ICL Testing - Progress Report

**日期**: 2025-11-13  
**分支**: feat/finetune-encoder  
**狀態**: ICL 測試進行中 ✅

---

## 📊 測試配置

### 資料集
- **來源**: Optical dataset (examples/optical/)
- **測試集**: 347 samples (10% of 3456 total)
- **訓練集**: 2764 samples (80%) - 用於選擇 prompt examples
- **驗證集**: 345 samples (10%)

### 測試參數
- **Shot 數**: 1, 3, 5
- **每個 shot 測試樣本數**: 10 samples
- **模型**: MiMo-Audio-7B-Base
- **Tokenizer**: MiMo-Audio-Tokenizer

---

## ✅ 已完成

### 1. 快速驗證測試
- **配置**: 1-shot, 3 samples
- **結果**: 3/3 成功 (100%)
- **時間**: ~10 秒
- **輸出**: `outputs/optical_icl_quick_test/1shot/`

**生成的檔案**:
- `113_1shot_enhanced.wav` (721KB) - 輸入: girl6_WOLDV_113.wav
- `262_1shot_enhanced.wav` (353KB) - 輸入: girl4_WOLDV_262.wav
- `005_1shot_enhanced.wav` (151KB) - 輸入: girl1_WOLDV_005.wav

### 2. 批次測試
- **狀態**: 進行中 🔄
- **配置**: 1, 3, 5-shot × 10 samples each
- **預計完成時間**: ~5-10 分鐘

---

## 🎯 ICL 測試方法

### Prompt Examples 格式
```python
{
    "input_audio": "噪聲音訊路徑",
    "output_audio": "乾淨音訊路徑", 
    "output_transcription": "文字標註"
}
```

### Instruction 格式
```python
instruction = f"請增強這段語音，內容是：{text}"
```

### API 呼叫
```python
text_output = model.in_context_learning_s2s(
    instruction,
    prompt_examples,
    input_audio,
    max_new_tokens=8192,
    output_audio_path=output_path
)
```

---

## 📈 觀察到的模式

### Text Channel 輸出
測試過程中，模型的 text channel 輸出了以下內容：

1. **重複 prompt**: 
   ```
   請增強這段語音，內容是：大家對他的演講很滿意
   ```

2. **內容識別**:
   ```
   他今天覺得頭轟轟層層的
   ```

3. **描述性輸出**:
   ```
   他對這項工作很有興趣，所以做得很好。
   ```

4. **混合輸出**:
   ```
   這個人看起來彬彬有禮，大家對他的演講很滿意。
   ```

**觀察**: 模型能夠識別語音內容，但輸出格式不太穩定，有時重複指令，有時擴展描述。

---

## 🔍 下一步計畫

### 短期 (今天)
1. ✅ 完成 1, 3, 5-shot 批次測試
2. ⏳ 分析不同 shot 數的效果差異
3. ⏳ 聆聽生成的音訊品質
4. ⏳ 與原始音訊和乾淨音訊比較

### 中期 (本週)
1. 測試更多 shots (10, 20)
2. 測試不同的 instruction 格式
3. 評估音訊品質指標 (PESQ, STOI 等)
4. 決定是否需要微調

### 長期目標
**選項 A**: 如果 ICL 效果已經很好
- 優化 prompt examples 選擇策略
- 調整 instruction 格式
- 部署應用

**選項 B**: 如果 ICL 效果不足
- 完成 encoder 微調腳本
- 執行完整微調訓練
- 比較微調前後效果

---

## 📂 檔案結構

```
MiMo-Audio/
├── scripts/data_management/
│   └── test_icl_optical.py          # ICL 測試腳本
├── batch_test_optical_icl.sh        # 批次測試啟動腳本
├── data/
│   ├── manifests/
│   │   └── optical_manifest.json    # 完整資料集 manifest
│   └── splits/finetune_optical/
│       ├── train.json               # 2764 samples
│       ├── val.json                 # 345 samples
│       └── test.json                # 347 samples
└── outputs/
    ├── optical_icl_quick_test/      # 快速驗證結果
    │   └── 1shot/
    │       ├── *_1shot_enhanced.wav
    │       └── results.json
    └── optical_icl_test/            # 批次測試結果 (進行中)
        ├── 1shot/
        ├── 3shot/
        └── 5shot/
```

---

## 💡 技術筆記

### 遇到的問題與解決
1. **API 方法名稱**: 
   - ❌ `model.inference()` 
   - ✅ `model.in_context_learning_s2s()`

2. **參數名稱**:
   - ❌ `input_audio=...` (keyword)
   - ✅ 位置參數: `instruction, prompt_examples, audio, ...`

3. **Prompt 欄位名稱**:
   - ❌ `"text": "..."`
   - ✅ `"output_transcription": "..."`

### 成功的配置
```python
# 正確的 API 呼叫
model.in_context_learning_s2s(
    instruction,              # str: 任務指令
    prompt_examples,          # list: 示例列表
    input_audio,             # str: 輸入音訊路徑
    max_new_tokens=8192,     # int: 最大生成 token 數
    output_audio_path=path   # str: 輸出路徑
)
```

---

**最後更新**: 2025-11-13 12:50 UTC
**狀態**: 批次測試進行中，預計 5-10 分鐘完成
