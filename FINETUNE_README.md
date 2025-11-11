# MiMo-Audio Encoder Fine-tuning

針對嚴重噪聲場景微調 MiMo-Audio 的 Encoder。

## 📋 概述

本分支專注於使用 **LoRA (Low-Rank Adaptation)** 方法微調 MiMo-Audio-Tokenizer 的 Encoder 部分，以提升對嚴重噪聲/損壞音訊的處理能力。

### 為什麼微調 Encoder？

1. **輸入品質問題**: 當輸入音訊嚴重損壞或噪聲很大時，Encoder 可能無法提取有效特徵
2. **針對性優化**: 通過微調 Encoder，讓它學會從噪聲音訊中提取更魯棒的特徵
3. **效率**: 使用 LoRA 只需訓練少量參數（~1-5%），大大降低訓練成本

## 🗂️ 檔案結構

```
feat/finetune-encoder/
├── FINETUNE_README.md              # 本檔案
├── FINETUNING_GUIDE.md             # 詳細的微調指南
├── finetune_encoder.py             # 主要訓練腳本
├── finetune_input_local.py         # 輸入處理工具
├── prepare_training_data.py        # 資料準備腳本
└── reproduce_exp_d2.py             # 重現 exp(D2) 實驗
```

## 🚀 快速開始

### 1. 準備訓練資料

假設您有一個資料目錄，包含：
- `mix/`: 噪聲音訊
- `spk1/`: 對應的乾淨音訊

```bash
# 準備訓練/驗證資料集
python prepare_training_data.py \
    --source_dir ./examples/ldv \
    --output_dir ./data/finetune_ldv \
    --train_ratio 0.8
```

這會創建：
```
data/finetune_ldv/
├── train/
│   ├── mix/          # 訓練集噪聲音訊
│   ├── spk1/         # 訓練集乾淨音訊
│   └── train_pairs.json
├── val/
│   ├── mix/          # 驗證集噪聲音訊
│   ├── spk1/         # 驗證集乾淨音訊
│   └── val_pairs.json
└── dataset_stats.json
```

### 2. 開始訓練

```bash
# 基礎訓練
python finetune_encoder.py \
    --data_dir ./data/finetune_ldv/train \
    --val_data_dir ./data/finetune_ldv/val \
    --tokenizer_path ./models/MiMo-Audio-Tokenizer \
    --output_dir ./outputs/finetune_encoder_ldv \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4
```

### 3. 監控訓練

訓練過程會保存：
- `outputs/finetune_encoder_ldv/config.json`: 訓練配置
- `outputs/finetune_encoder_ldv/best_model/`: 最佳模型
- `outputs/finetune_encoder_ldv/checkpoint_epoch_*/`: 定期檢查點

## 📊 訓練策略

### LoRA 參數建議

| 場景 | rank | alpha | 訓練參數量 |
|------|------|-------|-----------|
| 輕量級 | 4 | 8 | ~0.5% |
| 標準 | 8 | 16 | ~1% |
| 重度 | 16 | 32 | ~2-3% |

### 學習率建議

- **初始 LR**: 1e-4 到 5e-4
- **Warmup**: 100-500 steps
- **調度器**: Linear decay with warmup

### 批次大小

- RTX 5090 (24GB): batch_size=8-16
- RTX 4090 (24GB): batch_size=4-8
- 更小的 GPU: batch_size=2-4

## 🔧 高級配置

### 只微調特定層

修改 `finetune_encoder.py` 中的 `apply_lora_to_encoder()` 函數：

```python
# 只對最後幾層應用 LoRA
for name, module in encoder.named_modules():
    if 'layer_10' in name or 'layer_11' in name:  # 只微調最後兩層
        # 應用 LoRA
        ...
```

### 凍結/解凍策略

```python
# 先凍結所有層訓練幾個 epoch
for epoch in range(5):
    train_epoch(...)

# 然後解凍更多層繼續訓練
unfreeze_layers(encoder, layers_to_unfreeze=['layer_8', 'layer_9', 'layer_10'])
```

## 📈 評估

訓練完成後，使用微調後的 Tokenizer 進行推理：

```bash
# 使用微調後的 Tokenizer
python inference_with_finetuned.py \
    --input_audio noisy_audio.wav \
    --output_audio enhanced_audio.wav \
    --tokenizer_path ./outputs/finetune_encoder_ldv/best_model
```

## 📚 相關資源

- **詳細指南**: 參考 `FINETUNING_GUIDE.md`
- **LoRA 論文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **MiMo-Audio**: [原始 Repository](https://github.com/Xiaomi-Mimo-audio)

## ⚠️ 注意事項

1. **目前狀態**: `finetune_encoder.py` 是一個框架腳本，需要根據實際的 MiMo-Audio API 完成實現
2. **主要待實現部分**:
   - 載入 Audio Tokenizer 模型
   - 音訊載入和預處理
   - Encoder 的前向傳播
   - 完整的訓練循環

3. **資料格式**: 確保訓練資料的採樣率、通道數等與預訓練模型一致

## 🤝 貢獻

如果您完善了訓練腳本或發現了更好的訓練策略，歡迎提交 PR！

## 📝 實驗記錄

記錄您的實驗結果：

```markdown
### Experiment 1 - LDV Dataset
- Date: 2025-11-11
- Data: LDV boy1 speaker (828 pairs)
- Config: rank=8, alpha=16, lr=1e-4
- Results: TBD
```

