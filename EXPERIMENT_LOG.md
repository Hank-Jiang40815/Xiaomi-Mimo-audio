# MiMo-Audio 實驗記錄

## 最新實驗 (2025-11-18)
### 🧪 Encoder Fine-tuning V2 - Codebook Alignment + Mel Normalization
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 訓練：`bash start_training_v2.sh`（已啟用 `--normalize-mel`）→ docker + tmux `finetune_v2`，輸出 `outputs/optical_lora_v2_norm_r32_e100/`
- 推論：`CHECKPOINT=outputs/optical_lora_v2_norm_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2_norm/enhanced_050.wav ./test_inference_docker.sh --no-tmux`

**關鍵結果**:
- Train Loss: 6.06 → 0.063；Val Loss: 3.14 → 0.040（最佳 0.033 @ epoch 47）
- Mel per-band z-score 正規化讓 loss 尺度下降、收斂更快；最佳點出現在中段
- Checkpoints：`best_model.pt` + 每 10 epoch 快照；新 log `training_history.json`
- 推論樣本：`outputs/test_inference_v2_norm/enhanced_050.wav`（另存 `examples/codebook_align_inference/boy1_WOLDV_050_enhanced_v2_norm.wav`）

**洞察 / 待辦**:
1. Normalize 版本顯示 loss 下降幅度明顯，需配合 SI-SDR/PESQ 驗證是否真的改善音質
2. 建議使用同一 script 批次產線：`CHECKPOINT=outputs/optical_lora_v2_norm_r32_e100/best_model.pt bash test_inference_quick.sh`
3. 下一步：修補 splits＋加入 early stopping，再重訓比較 norm vs non-norm 表現

---

## 過往實驗 (2025-11-17)
### 🧪 Encoder Fine-tuning V2 - Codebook Alignment (LoRA Rank 32)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 訓練：`bash start_training_v2.sh` → docker + tmux，輸出於 `outputs/optical_lora_v2_r32_e100/`
- 推論：`CHECKPOINT=outputs/optical_lora_v2_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2/enhanced_050.wav ./test_inference_docker.sh --no-tmux`

**關鍵結果**:
- Loss 組成：Feature MSE + Codebook L1 + VQ commit（`lambda_feat=1, lambda_code=1, lambda_vq=0.1`）
- Train Loss: 12.08 → 8.22；Val Loss: 9.23 → 8.20（最佳 8.06 @ epoch 2）
- Checkpoints：`best_model.pt` + 每 10 epoch 快照；訓練曲線 `training_history.json`
- 推論樣本：`outputs/test_inference_v2/enhanced_050.wav`（與 `examples/optical/mix/boy1_WOLDV_050.wav` 對照）

**洞察 / 待辦**:
1. Codebook-aware Loss 可穩定訓練，但仍需檢查資料洩漏對 Val 的影響
2. 建議以 `test_inference_docker.sh` 批次生成樣本並計算 SI-SDR/PESQ
3. 若要縮短訓練，可加入 early stopping（最佳點依舊落在前幾個 epoch）

---

## 過往實驗 (2025-11-13)
### 🧪 Encoder Fine-tuning - LoRA Rank 32, 100 Epochs
**狀態**: ✅ 已完成  
**詳細報告**: [FINETUNE_EXPERIMENT_R32_E100.md](FINETUNE_EXPERIMENT_R32_E100.md)

**核心發現**:
- ✅ Training 成功完成 100 epochs (33.51% loss 改善)
- ✅ 無過擬合 (Train-Val gap 僅 0.28%)
- ⚠️ 最佳 Val loss 在 Epoch 2 出現，100 epochs 過多
- ⚠️ 發現資料集洩漏問題 (196/136/208 樣本重疊)

**後續行動**:
1. 修復資料洩漏，重新生成 splits
2. 用乾淨資料集重訓，加入 early stopping
3. 客觀評估音訊品質 (SI-SDR, PESQ)

---

## Docker 實驗 (2025-10-13)

## 實驗目的
驗證 MiMo-Audio 模型在 Docker 容器中的完整功能，包括：
1. SFT (Supervised Fine-Tuning) 模型的多種任務能力
2. Base 預訓練模型的 In-Context Learning 能力

## 環境配置

### Docker 環境
- **Base Image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.12
- **PyTorch**: 2.6.0
- **TorchAudio**: 2.6.0
- **CUDA**: 12.1
- **Flash Attention**: 2.7.4.post1

### 硬體需求
- GPU with CUDA support
- 至少 16GB VRAM（用於 7B 模型）
- 約 40GB 磁碟空間（包含兩個模型）

## 實驗一：SFT 模型測試

### 模型
- **模型名稱**: MiMo-Audio-7B-Instruct
- **模型大小**: ~15GB
- **模型類型**: 指令微調模型

### 測試腳本
`inference_example_sft.py`

### 測試功能

#### 1. 基本 TTS (Text-to-Speech)
- **輸入文本**: "今天天气真好"
- **輸出檔案**: `examples/tts.wav` (233 KB)
- **結果**: ✅ 成功生成自然流暢的中文語音

#### 2. 指令式 TTS
- **輸入文本**: "今天天气真好"
- **指令**: "用小孩子的声音开心的说"
- **輸出檔案**: `examples/instruct_tts.wav` (106 KB)
- **結果**: ✅ 成功模擬小孩子的聲音特徵和情緒

#### 3. 自然指令 TTS
- **輸入文本**: "用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！"
- **輸出檔案**: `examples/natural_instruction_tts.wav` (106 KB)
- **結果**: ✅ 成功表現氣喘吁吁的效果

#### 4. 音頻理解
- **輸入音頻**: `examples/spoken_dialogue_assistant_turn_1.wav`
- **任務**: "Summarize the audio."
- **結果**: ✅ 成功總結音頻內容（英文摘要）

#### 5. 音頻理解（帶思考過程）
- **輸入音頻**: `examples/spoken_dialogue_assistant_turn_1.wav`
- **任務**: "Summarize the audio."
- **結果**: ✅ 提供詳細的思考過程和分析

#### 6. 多輪口語對話
- **輸入**: 
  - 第一輪：音頻問天氣
  - 第二輪：回答「北京」
- **輸出檔案**: `examples/spoken_dialogue_assistant_turn_2.wav` (1.4 MB)
- **結果**: ✅ 成功生成連貫的多輪對話音頻

#### 7. 語音轉文字對話
- **輸入**: 多輪語音和文字混合對話
- **結果**: ✅ 成功處理語音輸入並生成文字回應

#### 8. 純文字對話
- **任務**: 介紹北京的旅遊景點
- **結果**: ✅ 生成詳細的旅遊推薦（包含景點、交通、門票等資訊）

### 性能指標
- **模型載入時間**: ~8 秒
- **Tokenizer 載入時間**: ~11 秒
- **單次推理時間**: 視任務複雜度而定（2-30 秒）

## 實驗二：Base 模型 In-Context Learning 測試

### 模型
- **模型名稱**: MiMo-Audio-7B-Base
- **模型大小**: ~15GB
- **模型類型**: 預訓練基礎模型

### 測試腳本
`inference_example_pretrain.py`

### 測試任務：音色轉換 (Voice Conversion)

#### 實驗設計
- **任務**: 將說話人 0013 的聲音轉換為說話人 0019 的音色
- **方法**: In-Context Learning (Few-shot Learning)
- **示例數量**: 5 對語音轉換示例

#### 示例對（說話人 0013 → 0019）
1. `0013_000139.wav` → `0019_000139.wav`: "Cuckoos is downheaded and crying."
2. `0013_000963.wav` → `0019_000963.wav`: "She said in subdued voice."
3. `0013_000559.wav` → `0019_000559.wav`: "A raging fire was-in his eyes."
4. `0013_001142.wav` → `0019_001142.wav`: "Does the one that wins get the crowned?"
5. `0013_000769.wav` → `0019_000769.wav`: "Not much use is it, sam?"

#### 測試輸入
- **輸入音頻**: `examples/ESD/0013_000200.wav` (說話人 0013)
- **指令**: "Convert the timbre of the input speech to target timbre."

#### 測試結果
- **輸出檔案**: `examples/in_context_learning_s2s.wav` (136 KB)
- **轉錄文本**: "The cloth does not look worth much."
- **結果**: ✅ 成功將音色從說話人 0013 轉換為說話人 0019

### 性能指標
- **模型載入時間**: ~8 秒
- **Tokenizer 載入時間**: ~10 秒
- **推理時間**: 包含處理 5 個示例對 + 生成輸出
- **max_new_tokens**: 8192

## 實驗結論

### 成功驗證的功能
1. ✅ Docker 環境配置正確，GPU 加速正常
2. ✅ PyTorch 2.6.0 與 CUDA 12.1 相容性良好
3. ✅ SFT 模型支援多種任務（TTS、對話、理解）
4. ✅ Base 模型支援 In-Context Learning
5. ✅ 模型載入和推理穩定可靠
6. ✅ 音頻生成質量良好

### 兩種模型的差異
| 特性 | Base (Pretrain) | Instruct (SFT) |
|------|-----------------|----------------|
| **使用方式** | Few-shot examples | 直接指令 |
| **靈活性** | 高（可學習新任務） | 中（預定義任務） |
| **易用性** | 低（需準備示例） | 高（直接調用） |
| **適用場景** | 研究探索 | 生產應用 |

## 如何重現實驗

### 前置準備

#### 1. 下載模型
```bash
# 下載 Instruct 模型（用於實驗一）
huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --local-dir models/MiMo-Audio-7B-Instruct

# 下載 Base 模型（用於實驗二）
huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Base \
  --local-dir models/MiMo-Audio-7B-Base

# 下載 Tokenizer
huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir models/MiMo-Audio-Tokenizer
```

#### 2. 構建 Docker 映像
```bash
docker build -t mimo-audio:latest .
```

### 重現實驗一：SFT 模型測試

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/examples:/app/examples \
  mimo-audio:latest \
  python3.12 inference_example_sft.py
```

**預期輸出檔案**:
- `examples/tts.wav`
- `examples/instruct_tts.wav`
- `examples/natural_instruction_tts.wav`
- `examples/spoken_dialogue_assistant_turn_2.wav`

### 重現實驗二：Base 模型 In-Context Learning

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/examples:/app/examples \
  mimo-audio:latest \
  python3.12 inference_example_pretrain.py
```

**預期輸出檔案**:
- `examples/in_context_learning_s2s.wav`

### 快速測試（使用 smoke test）

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  mimo-audio:latest \
  python3.12 scripts/smoke_test.py \
  --audio ./examples/prompt_speech_zh_m.wav
```

## 生成的檔案清單

### 實驗一生成的檔案
- `examples/tts.wav` - 基本 TTS 輸出
- `examples/instruct_tts.wav` - 指令式 TTS（小孩子聲音）
- `examples/natural_instruction_tts.wav` - 自然指令 TTS（氣喘吁吁）
- `examples/spoken_dialogue_assistant_turn_2.wav` - 多輪對話第二輪回應

### 實驗二生成的檔案
- `examples/in_context_learning_s2s.wav` - 音色轉換結果

### Smoke Test 生成的檔案
- `outputs/tts_sample.wav` - 測試語音樣本

## 問題排查

### 常見問題

**Q1: 遇到 CUDA out of memory 錯誤**
```
A: 需要至少 16GB VRAM。可以嘗試：
   - 減少 batch size
   - 使用更小的模型
   - 關閉其他 GPU 程序
```

**Q2: 模型載入時間過長**
```
A: 正常現象。7B 模型需要 7-10 秒載入時間。
   可以使用模型緩存加速後續載入。
```

**Q3: 音頻質量不佳**
```
A: 檢查：
   - 輸入音頻質量
   - 採樣率是否正確
   - 是否使用了正確的模型
```

## 技術細節

### Docker 映像層級
1. NVIDIA CUDA 12.1.1 runtime base
2. Python 3.12 安裝
3. PyTorch 2.6.0 + TorchAudio 2.6.0
4. Flash Attention 2.7.4
5. 專案依賴安裝

### 關鍵依賴版本
- `torch==2.6.0`
- `torchaudio==2.6.0`
- `transformers==4.49.0`
- `accelerate>=1.9.0`
- `librosa>=0.11.0`
- `gradio==5.46.1`

## 未來改進方向

1. **性能優化**
   - 實現模型量化（INT8/INT4）
   - 支援批次推理
   - 優化記憶體使用

2. **功能擴展**
   - 支援更多語言
   - 增加實時推理能力
   - Web UI 整合

3. **測試擴展**
   - 增加自動化測試
   - 音頻品質評估指標
   - 效能基準測試

## 參考資料
- MiMo-Audio GitHub: https://github.com/XiaomiMiMo/MiMo-Audio
- Hugging Face Models: https://huggingface.co/XiaomiMiMo

## 實驗三：optical 正規化後的初步降噪（noisereduce）

### 目的
- `examples/optical_normalized/{mix,spk1}` 仍可聽見高頻砂點/顆粒噪聲；以 noisereduce 先做保守降噪，供 few‑shot 與主觀評測使用。

### 流程與參數
- 腳本：`scripts/denoise_optical.py`
- 來源與輸出：
  - 來源：`examples/optical_normalized/{mix,spk1}`
  - 輸出：`examples/optical_denoised/{mix,spk1}`（mirror 同名）
- 前處理濾波：`highpass=90 Hz`、`lowpass=8500 Hz`
- 降噪：noisereduce 非平穩（`stationary=False`）、`prop_decrease=0.85`
- 平滑：`time_mask_smooth_ms=64`、`freq_mask_smooth_hz=150`
- 後處理：峰值正規化 `-1.0 dBFS`，輸出 `WAV PCM16`

### 執行（Docker）
```bash
docker build -t mimo-audio:latest .
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/app -w /app mimo-audio:latest \
  python -u scripts/denoise_optical.py
```

### 結果摘要（2025-11-06）
- 檔案數：
  - mix：3456 個
  - spk1：3456 個
- 容量：
  - `examples/optical_denoised/mix`：435 MB
  - `examples/optical_denoised/spk1`：435 MB
- 備註：輸出檔依 `.gitignore` 規則忽略版本控管。
