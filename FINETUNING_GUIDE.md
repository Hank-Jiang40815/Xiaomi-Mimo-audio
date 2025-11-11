# MiMo-Audio 微調指南

本指南提供多種微調 MiMo-Audio 模型的方法，適用於音訊增強、語音轉換等任務。

## 📋 目錄

1. [微調方法概覽](#微調方法概覽)
2. [環境準備](#環境準備)
3. [資料準備](#資料準備)
4. [方法 1: LoRA 微調（推薦）](#方法-1-lora-微調推薦)
5. [方法 2: Full Fine-tuning](#方法-2-full-fine-tuning)
6. [方法 3: Adapter 微調](#方法-3-adapter-微調)
7. [評估與推論](#評估與推論)

---

## 微調方法概覽

### 基礎方法

| 方法 | 記憶體需求 | 訓練速度 | 性能 | 適用場景 |
|------|-----------|---------|------|---------|
| **LoRA** | 🟢 低 (16-24GB) | 🟢 快 | 🟡 良好 | 小資料集、資源受限 |
| **Full Fine-tuning** | 🔴 高 (80GB+) | 🔴 慢 | 🟢 最佳 | 大資料集、充足資源 |
| **Adapter** | 🟡 中 (32-48GB) | 🟡 中等 | 🟡 良好 | 平衡效能與成本 |

### 針對嚴重噪聲/缺損信號的特殊策略 ⭐

| 策略 | 目標組件 | 記憶體需求 | 效果 | 適用場景 |
|------|---------|-----------|------|---------|
| **Encoder-Focused LoRA** | Audio Tokenizer Encoder | 🟢 低 (12-18GB) | 🟢 優秀 | **嚴重噪聲、信號缺損** |
| **Hierarchical Fine-tuning** | Encoder → Decoder → LLM | 🟡 中 (20-32GB) | 🟢 最佳 | 嚴重噪聲 + 語義理解 |
| **Denoising Pre-training** | 全模型（兩階段） | 🟡 中 (24-40GB) | 🟢 優秀 | 極端噪聲環境 |

---

## 環境準備

### 1. 安裝微調依賴

```bash
# 安裝基礎微調套件
pip install -r finetune_requirements.txt

# 或手動安裝
pip install peft==0.12.0
pip install bitsandbytes==0.44.1
pip install accelerate==0.34.2
pip install deepspeed==0.15.4
pip install wandb==0.18.5  # 可選，用於訓練監控
```

### 2. 驗證 GPU 可用性

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 資料準備

### 資料格式

MiMo-Audio 微調需要配對的音訊資料：

```
dataset/
├── train/
│   ├── noisy/          # 輸入音訊（有噪音）
│   │   ├── audio_001.wav
│   │   └── ...
│   └── clean/          # 目標音訊（乾淨）
│       ├── audio_001.wav
│       └── ...
└── val/
    ├── noisy/
    └── clean/
```

### 範例：使用 LDV 資料集

```python
# 資料集結構
{
    "train": [
        {
            "noisy_audio": "examples/ldv/mix/boy1_papercup_LDV_001.wav",
            "clean_audio": "examples/ldv/spk/boy1_papercup_clean_001.wav",
            "text": "他一口氣喝了三碗豆漿"  # 可選
        },
        ...
    ]
}
```

---

## 🎯 針對嚴重噪聲/缺損信號的專用方法

### 問題分析

當輸入信號有**嚴重噪聲或缺損**時：
- ❌ 標準 LoRA 主要微調 LLM 層，對噪聲信號特徵提取不足
- ❌ 噪聲在 Encoder 階段就被編碼，後續難以修復
- ✅ **應該重點微調 Audio Tokenizer 的 Encoder 部分**

### 方法 1-A: Encoder-Focused LoRA ⭐ 推薦用於噪聲信號

這是針對您的 LDV 嚴重噪聲場景的**最佳方案**！

#### 原理
```
輸入噪聲音訊 → [Encoder 強化] → 乾淨特徵 → Decoder → 輸出
                    ↑ 微調重點
```

#### 配置範例

```bash
python finetune_lora.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --train_data examples/ldv \
    --output_dir outputs/encoder_focused \
    --target_modules "encoder" \
    --finetune_encoder_only \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_epochs 5 \
    --batch_size 4
```

#### 進階配置：分層微調

```bash
# 階段 1: 只微調 Encoder（學習噪聲特徵）
python finetune_lora.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --train_data examples/ldv \
    --output_dir outputs/stage1_encoder \
    --target_modules "encoder.layers" \
    --freeze_decoder \
    --freeze_llm \
    --lora_r 32 \
    --num_epochs 5

# 階段 2: 微調 Decoder（學習重建）
python finetune_lora.py \
    --model_path outputs/stage1_encoder \
    --train_data examples/ldv \
    --output_dir outputs/stage2_decoder \
    --target_modules "decoder.layers" \
    --freeze_encoder \
    --freeze_llm \
    --lora_r 16 \
    --num_epochs 3

# 階段 3: 端到端微調（可選）
python finetune_lora.py \
    --model_path outputs/stage2_decoder \
    --train_data examples/ldv \
    --output_dir outputs/stage3_full \
    --lora_r 8 \
    --num_epochs 2
```

#### 優點
- ✅ 直接針對噪聲特徵提取
- ✅ 記憶體需求低（12-18GB）
- ✅ 對嚴重噪聲效果最好
- ✅ 適合您的 LDV x65、x70 場景

---

## 方法 1-B: 標準 LoRA 微調

### 優點
- ✅ 記憶體需求低（RTX 4090 24GB 可運行）
- ✅ 訓練快速
- ✅ 支援多任務切換
- ✅ 容易合併到原模型

### 適用場景
- 輕度噪聲
- 語義層面的改進
- 風格轉換

### 快速開始

```bash
# 使用預設設定微調
python finetune_lora.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --train_data dataset/train \
    --val_data dataset/val \
    --output_dir outputs/lora_audio_enhancement \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

### 進階配置

```bash
python finetune_lora.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --train_data dataset/train \
    --val_data dataset/val \
    --output_dir outputs/lora_advanced \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules "q_proj,v_proj,k_proj,o_proj" \
    --use_8bit \
    --wandb_project "mimo-audio-finetune"
```

### LoRA 參數說明

- `lora_r`: LoRA 秩（8-64），越高越接近全參數微調
- `lora_alpha`: 縮放係數，通常設為 `2 * lora_r`
- `lora_dropout`: Dropout 率，防止過擬合
- `target_modules`: 要應用 LoRA 的模組

### 針對不同噪聲程度的模組選擇建議

| 噪聲程度 | 推薦目標模組 | LoRA Rank | 原因 |
|---------|------------|-----------|------|
| **嚴重** (LDV, x65, x70) | `encoder.layers` | 32-64 | 需要強化特徵提取 |
| **中度** | `encoder.layers,decoder.layers` | 16-32 | 提取+重建雙管齊下 |
| **輕度** | `q_proj,v_proj,k_proj,o_proj` | 8-16 | 標準 attention 微調 |

---

## 方法 1-C: Denoising Pre-training（極端噪聲）

針對**極端噪聲場景**的兩階段訓練策略：

### 階段 1: 降噪預訓練

```bash
# 使用對比學習訓練 Encoder 識別噪聲模式
python finetune_denoising.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --noisy_data examples/ldv/mix \
    --clean_data examples/ldv/spk \
    --output_dir outputs/denoising_pretrain \
    --objective "contrastive" \
    --num_epochs 10
```

### 階段 2: 任務微調

```bash
# 在降噪預訓練的基礎上進行任務微調
python finetune_lora.py \
    --model_path outputs/denoising_pretrain \
    --train_data examples/ldv \
    --output_dir outputs/denoising_finetune \
    --lora_r 16
```

---

## 方法 2: Full Fine-tuning

### 系統需求
- GPU: A100 80GB 或多卡並行
- RAM: 128GB+
- 建議使用 DeepSpeed ZeRO-3

### 使用 DeepSpeed

```bash
# 創建 DeepSpeed 配置檔
cat > ds_config.json <<EOF
{
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
EOF

# 執行訓練
deepspeed --num_gpus=2 finetune_full.py \
    --deepspeed ds_config.json \
    --model_path models/MiMo-Audio-7B-Base \
    --train_data dataset/train \
    --output_dir outputs/full_finetune
```

---

## 方法 3: Adapter 微調

介於 LoRA 和全參數微調之間的選擇。

```bash
python finetune_adapter.py \
    --model_path models/MiMo-Audio-7B-Base \
    --tokenizer_path models/MiMo-Audio-Tokenizer \
    --train_data dataset/train \
    --adapter_hidden_size 512 \
    --num_adapter_layers 4 \
    --output_dir outputs/adapter_finetune
```

---

## 評估與推論

### 1. 載入微調後的模型

```python
from src.mimo_audio.mimo_audio import MimoAudio
from peft import PeftModel

# 載入基礎模型
base_model = MimoAudio(
    model_path="models/MiMo-Audio-7B-Base",
    tokenizer_path="models/MiMo-Audio-Tokenizer"
)

# 載入 LoRA 權重
model = PeftModel.from_pretrained(
    base_model.model,
    "outputs/lora_audio_enhancement/checkpoint-best"
)
```

### 2. 推論測試

```python
# 測試音訊增強
result = base_model.audio_enhancement(
    input_audio="test_audio.wav",
    instruction="Enhance the audio quality and remove noise."
)
result.save("enhanced_output.wav")
```

### 3. 評估指標

```bash
python evaluate_finetune.py \
    --model_path outputs/lora_audio_enhancement/checkpoint-best \
    --test_data dataset/test \
    --metrics "pesq,si-sdr,stoi"
```

---

## 🔧 故障排除

### 1. CUDA Out of Memory

```bash
# 減小 batch size
--batch_size 1 --gradient_accumulation_steps 8

# 使用 8-bit 量化
--use_8bit

# 使用 gradient checkpointing
--gradient_checkpointing
```

### 2. 訓練不收斂

- 降低 learning rate: `1e-5` 至 `5e-5`
- 增加 warmup steps: `--warmup_steps 500`
- 檢查資料品質

### 3. 過擬合

- 增加 dropout: `--lora_dropout 0.1`
- 使用 weight decay: `--weight_decay 0.01`
- 增加訓練資料

### 4. 噪聲信號處理效果不佳 ⚠️

**症狀**: 輸出音訊仍有明顯噪聲或失真

**原因分析**:
- ❌ 只微調了 LLM 層，Encoder 沒有學習到噪聲特徵
- ❌ 噪聲在編碼階段就混入特徵，後續無法分離

**解決方案**:
```bash
# 方案 1: 改用 Encoder-Focused LoRA
--target_modules "encoder.layers" \
--freeze_decoder \
--freeze_llm

# 方案 2: 增加 Encoder 的 LoRA rank
--encoder_lora_r 64 \  # Encoder 用更高的 rank
--decoder_lora_r 16    # Decoder 用較低的 rank

# 方案 3: 使用分層訓練策略
# 先訓練 Encoder → 再訓練 Decoder → 最後端到端微調
```

### 5. x65 vs x70 效果差異大

**現象**: 不同噪聲級別的處理效果不一致

**解決方案**:
```bash
# 方案 1: 噪聲級別條件化訓練
--noise_level_conditioning \
--noise_levels "x60,x65,x70,xnonoise"

# 方案 2: 分別訓練不同噪聲級別的 LoRA
python finetune_lora.py --train_data ldv_x65 --output_dir lora_x65
python finetune_lora.py --train_data ldv_x70 --output_dir lora_x70

# 推論時根據噪聲級別切換 LoRA
```

---

## 📊 訓練監控

### 使用 Weights & Biases

```bash
# 安裝 wandb
pip install wandb
wandb login

# 訓練時啟用
python finetune_lora.py \
    --wandb_project "mimo-audio-finetune" \
    --wandb_run_name "lora-r16-lr1e4" \
    ...
```

### 使用 TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir outputs/lora_audio_enhancement/logs

# 瀏覽器開啟
# http://localhost:6006
```

---

## 🎯 最佳實踐

1. **從小規模開始**: 先用 100-200 筆資料測試流程
2. **使用 validation set**: 監控過擬合
3. **保存 checkpoints**: 每個 epoch 都保存
4. **記錄實驗**: 使用 wandb 或詳細的 log 檔案
5. **對比基準**: 與原模型和傳統方法比較

---

## 📚 相關資源

- [MiMo-Audio 官方文檔](https://github.com/XiaomiMiMo/MiMo-Audio)
- [PEFT 文檔](https://huggingface.co/docs/peft)
- [LoRA 論文](https://arxiv.org/abs/2106.09685)
- [本專案實驗記錄](EXPERIMENT_LOG.md)

---

## 💡 範例應用場景

### 1. 音訊降噪增強
```bash
python finetune_lora.py \
    --task audio_enhancement \
    --train_data examples/ldv \
    --num_epochs 3
```

### 2. 語音風格轉換
```bash
python finetune_lora.py \
    --task voice_conversion \
    --speaker_embedding \
    --num_epochs 5
```

### 3. 語音編輯
```bash
python finetune_lora.py \
    --task speech_editing \
    --enable_text_guidance \
    --num_epochs 4
```

---

**更新日期**: 2025-11-10  
**版本**: 1.0.0
