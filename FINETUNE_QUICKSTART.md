# Fine-tuning Encoder 快速開始指南

> 💡 **完整除錯紀錄請參考**: [FINETUNE_ENCODER_DEBUGGING_LOG.md](./FINETUNE_ENCODER_DEBUGGING_LOG.md)

---

## 🚀 快速開始

### 1. 測試運行（小資料集）
```bash
docker run --gpus all --rm \
    -v "$(pwd)":/workspace -w /workspace \
    mimo-audio:latest python finetune_encoder.py \
    --train-split data/splits/finetune_optical_test/train.json \
    --val-split data/splits/finetune_optical_test/val.json \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --epochs 1 \
    --lora-rank 8 \
    --output-dir ./outputs/finetune_test
```

### 2. 正式訓練（完整資料集）
```bash
docker run --gpus all --rm \
    -v "$(pwd)":/workspace -w /workspace \
    mimo-audio:latest python finetune_encoder.py \
    --train-split data/splits/finetune_optical/train.json \
    --val-split data/splits/finetune_optical/val.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 10 \
    --lora-rank 16 \
    --output-dir ./outputs/optical_lora_r16 \
    --learning-rate 1e-4
```

---

## ⚙️ 關鍵參數說明

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `--batch-size` | 2-4 | 每個 GPU 的 batch size，根據記憶體調整 |
| `--gradient-accumulation-steps` | 2-4 | 有效 batch size = batch_size × accumulation_steps |
| `--lora-rank` | 8, 16, 32 | LoRA 秩，越大可訓練參數越多 |
| `--lora-alpha` | 16.0 | LoRA 縮放因子，通常設為 rank 的 2 倍 |
| `--learning-rate` | 1e-4 | 學習率，可以從 1e-4 到 5e-4 嘗試 |
| `--epochs` | 10-20 | 訓練輪數 |

---

## 📊 期望結果

### 訓練時
```
Epoch 1: 100%|██████████| 5/5 [00:00<00:00, 5.89it/s]
Train Loss: 14.27
Val Loss: 14.90
```

### 輸出檔案
```
outputs/optical_lora_r16/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
├── best_model.pt
└── training_log.txt
```

---

## ⚠️ 常見問題

### 問題 1: OOM (Out of Memory)
**解決方案**:
- 降低 `--batch-size` 到 1
- 減少 `--lora-rank` 到 8
- 增加 `--gradient-accumulation-steps`

### 問題 2: Flash Attention dtype 錯誤
```
RuntimeError: FlashAttention only support fp16 and bf16
```
**解決方案**: 已修復，模型自動轉換為 bfloat16

### 問題 3: 找不到 split 檔案
```
FileNotFoundError: Split file not found
```
**解決方案**: 確認資料分割檔案存在：
```bash
ls -la data/splits/finetune_optical/train.json
ls -la data/splits/finetune_optical/val.json
```

---

## 🎯 核心概念

### 輸入格式
- ✅ **Mel Spectrogram** (128 bands, hop=240, sr=24kHz)
- ❌ 不是原始波形 (waveform)

### 訓練策略
- **特徵匹配損失**: 讓噪音音訊的特徵接近乾淨音訊
- **LoRA 微調**: 只訓練 51.22% 的參數（661M / 1290M）
- **bfloat16**: 支援 Flash Attention，加速訓練

---

## 📚 相關文件

- [FINETUNE_ENCODER_DEBUGGING_LOG.md](./FINETUNE_ENCODER_DEBUGGING_LOG.md) - 完整除錯過程
- [FINETUNING_GUIDE.md](./FINETUNING_GUIDE.md) - 詳細說明
- [FINETUNE_README.md](./FINETUNE_README.md) - 概述

---

**最後更新**: 2025-11-13
