# Quick Start: Fine-tuning MiMo-Audio Encoder

**Branch**: `feat/finetune-encoder`  
**Ready to go!** ✅ All data and tools are prepared.

---

## 🎯 Dataset Ready

### Optical Dataset (推薦使用)
- **位置**: `examples/optical/`
- **檔案數**: 3456 pairs (100% 配對成功)
- **說明文字**: 100% 覆蓋率（Whisper 轉錄）
- **Manifest**: `data/manifests/optical_manifest.json`

```json
{
  "dataset_name": "optical",
  "total_samples": 3456,
  "matched_pairs": 3456,
  "samples": [
    {
      "id": "boy1_WOLDV_001",
      "speaker": "boy1",
      "noise_type": "WOLDV",
      "noisy_file": "examples/optical/mix/boy1_WOLDV_001.wav",
      "clean_file": "examples/optical/spk1/boy1_WOLDV_clean_001.wav",
      "text": "正學齊學校友書法比賽"
    },
    ...
  ]
}
```

---

## 🚀 Step 1: 準備訓練資料

使用 manifest 來建立 train/val splits：

```bash
# 使用 split_selector.py 建立 finetune splits
python scripts/data_management/split_selector.py \
    --manifest data/manifests/optical_manifest.json \
    --output-dir data/splits/finetune_optical \
    --mode finetune \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --random-seed 42
```

這會建立：
```
data/splits/finetune_optical/
├── train.json    # 80% (2764 samples)
├── val.json      # 10% (346 samples)
└── test.json     # 10% (346 samples)
```

或者使用 `prepare_training_data.py` 直接複製檔案：

```bash
python prepare_training_data.py \
    --source_dir ./examples/optical \
    --output_dir ./data/finetune_optical \
    --train_ratio 0.8
```

---

## 🔥 Step 2: 開始微調

### 基本訓練

```bash
python finetune_encoder.py \
    --data_dir ./data/finetune_optical \
    --output_dir ./outputs/optical_lora_r16 \
    --epochs 10 \
    --batch_size 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4
```

### 進階參數調整

```bash
python finetune_encoder.py \
    --data_dir ./data/finetune_optical \
    --output_dir ./outputs/optical_lora_r32 \
    --epochs 20 \
    --batch_size 8 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --save_steps 1000 \
    --eval_steps 500
```

### 使用 Docker 訓練（推薦）

```bash
# 啟動 MiMo-Audio Docker
docker run --gpus all -it --rm \
    -v /home/sbplab/Hank/MiMo-Audio:/workspace \
    mimo-audio:latest bash

# 在容器內執行訓練
cd /workspace
python finetune_encoder.py \
    --data_dir ./data/finetune_optical \
    --output_dir ./outputs/optical_lora_r16 \
    --epochs 10 \
    --batch_size 4 \
    --lora_r 16
```

---

## 📊 Step 3: 監控訓練

訓練過程會輸出：

```
Epoch 1/10
Step 100/691 | Loss: 0.1234 | LR: 1.00e-04
Step 200/691 | Loss: 0.0987 | LR: 1.00e-04
...
Validation Loss: 0.0876
Saving checkpoint to outputs/optical_lora_r16/checkpoint-1000
```

檢查輸出目錄：
```
outputs/optical_lora_r16/
├── checkpoint-1000/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── optimizer.pt
├── training_log.txt
└── validation_results.json
```

---

## 🧪 Step 4: 測試微調後的模型

### 使用測試集評估

```bash
python test_finetuned_encoder.py \
    --lora_path ./outputs/optical_lora_r16/checkpoint-1000 \
    --test_dir ./data/finetune_optical/val \
    --output_dir ./outputs/test_results
```

### 與 Baseline 比較

```bash
# 測試原始模型（無微調）
python experiment_audio_enhancement.py \
    --input_file examples/optical/mix/boy1_WOLDV_001.wav \
    --output_file examples/baseline_result.wav

# 測試微調模型
python experiment_audio_enhancement.py \
    --input_file examples/optical/mix/boy1_WOLDV_001.wav \
    --output_file examples/finetuned_result.wav \
    --lora_path ./outputs/optical_lora_r16/checkpoint-1000
```

---

## 📈 LoRA 參數說明

### `lora_r` (Rank)
- **作用**: LoRA 的秩，控制可訓練參數量
- **建議值**: 8, 16, 32
- **影響**: 
  - 越大 = 更多參數 = 更強表達能力，但訓練更慢
  - 越小 = 更快訓練 = 可能欠擬合

### `lora_alpha`
- **作用**: LoRA 的縮放因子
- **建議值**: 通常設為 `2 * lora_r`
- **影響**: 控制 LoRA 層的影響強度

### 參數量估算
- `lora_r=8`: ~0.5-1% 總參數
- `lora_r=16`: ~1-2% 總參數
- `lora_r=32`: ~2-4% 總參數

---

## 🎛️ 訓練技巧

### 1. 從小開始
```bash
# 先用小 rank 和少 epochs 測試
python finetune_encoder.py \
    --lora_r 8 \
    --epochs 5 \
    --batch_size 2
```

### 2. 逐步增大
```bash
# 如果效果好，增加 rank 和 epochs
python finetune_encoder.py \
    --lora_r 16 \
    --epochs 10 \
    --batch_size 4
```

### 3. 監控 Overfitting
- 觀察 train loss 和 val loss
- 如果 val loss 上升但 train loss 下降 → 過擬合
- 解決方法：
  - 減少 epochs
  - 增加 weight_decay
  - 使用 dropout

### 4. 使用 Text Annotations
```python
# 在訓練中加入文字資訊作為條件
# 可以參考 scripts/data_management/test_icl_with_text.py
```

---

## 🔍 Debug 建議

### 檢查資料載入
```bash
# 驗證 manifest 格式
python -c "import json; m=json.load(open('data/manifests/optical_manifest.json')); print(f'Total: {m[\"total_samples\"]}, Pairs: {m[\"matched_pairs\"]}')"
```

### 檢查模型載入
```bash
# 測試模型是否能正常載入
python -c "from src.mimo_audio_tokenizer import AudioTokenizer; tok = AudioTokenizer.from_pretrained('models/MiMo-Audio-Tokenizer'); print('Model loaded!')"
```

### 檢查 GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📚 相關文件

- `FINETUNE_README.md` - 詳細說明
- `FINETUNING_GUIDE.md` - 完整指南
- `FINETUNE_ARCHITECTURE_ANALYSIS.md` - 技術分析
- `data/DATA_MANAGEMENT_GUIDE.md` - 資料管理指南
- `BRANCH_STATUS.md` - 當前狀態總結

---

## ✅ Checklist

開始前確認：

- [ ] 在 `feat/finetune-encoder` 分支
- [ ] Optical dataset manifest 已準備好（3456 samples）
- [ ] Docker 環境可用（`mimo-audio:latest`）
- [ ] GPU 可用（`nvidia-smi` 檢查）
- [ ] 了解 LoRA 參數影響
- [ ] 準備好儲存空間（checkpoints 可能很大）

---

**準備開始了嗎？從 Step 1 開始！** 🚀

有問題隨時問我！
