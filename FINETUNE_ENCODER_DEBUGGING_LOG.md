# Fine-tuning Encoder 除錯紀錄

**日期**: 2025-11-13  
**任務**: 實現 MiMo-Audio Encoder 的 LoRA Fine-tuning  
**目標**: 針對 Optical 嚴重噪音資料集微調音訊編碼器

---

## 📋 除錯過程總覽

本次除錯共解決 **5 個關鍵問題**，花費約 2 小時，最終成功實現可運行的 fine-tuning pipeline。

---

## 🐛 問題 1: LoRA 注入時的 RuntimeError

### 錯誤訊息
```python
RuntimeError: dictionary changed size during iteration
File "/workspace/finetune_encoder.py", line 108, in inject_lora_to_encoder
    for name, module in encoder.named_modules():
```

### 問題原因
在遍歷 `encoder.named_modules()` 的同時，使用 `setattr()` 動態添加新的子模組（LoRA 層），導致 Python 的迭代器檢測到字典大小變化而拋出錯誤。

### 解決方案
```python
# ❌ 錯誤做法
for name, module in encoder.named_modules():
    if isinstance(module, nn.Linear):
        lora = LoRALayer(...)
        setattr(parent, f'{layer_name}_lora', lora)  # 修改字典

# ✅ 正確做法
modules_to_modify = []
for name, module in list(encoder.named_modules()):  # 轉換為 list
    if isinstance(module, nn.Linear):
        modules_to_modify.append((name, module))

for name, module in modules_to_modify:
    lora = LoRALayer(...)
    # 不使用 setattr，直接註冊參數到原始 module
    module.lora_A = lora.lora_A
    module.lora_B = lora.lora_B
```

### 學到的知識
- Python 迭代器在迭代過程中不允許修改集合大小
- `named_modules()` 返回的是一個生成器，需要先轉換為 list
- PyTorch 模組可以直接通過屬性賦值註冊參數

---

## 🐛 問題 2: MiMoAudioTokenizer.encode() API 參數錯誤

### 錯誤訊息
```python
TypeError: MiMoAudioTokenizer.encode() got an unexpected keyword argument 'input_features'
```

### 問題原因
混淆了兩個不同層級的 `encode()` 方法：
- `MiMoAudioTokenizer.encode()` 的第一個參數是 `mels`
- `AudioEncoder.encode()` 的第一個參數是 `input_features`

### 解決方案
```python
# ❌ 錯誤呼叫
model.encode(input_features=mels, input_lens=audio_lengths)

# ✅ 正確呼叫
model.encode(mels=mels, input_lens=audio_lengths)
# 或直接使用位置參數
model.encode(mels, input_lens=audio_lengths)
```

### 程式碼對照
```python
# MiMoAudioTokenizer (外層包裝器)
class MiMoAudioTokenizer:
    def encode(self, mels, input_lens, use_quantizer=True):
        ...
        return self.encoder.encode(mels, input_lens, use_quantizer)

# AudioEncoder (內部編碼器)
class AudioEncoder:
    def encode(self, input_features, input_lens, ...):
        ...
```

### 學到的知識
- MiMo-Audio 有兩層 API 包裝
- 需要仔細閱讀源碼確認參數名稱
- 使用 `grep_search` 查找函數定義

---

## 🐛 問題 3: 輸入資料格式錯誤（最複雜）

### 錯誤訊息
```python
RuntimeError: shape '[1, 240000, 240000]' is invalid for input of size 240000
RuntimeError: Given groups=1, weight of size [1280, 128, 3], expected input[1, 240000, 1] to have 128 channels
```

### 問題原因
**關鍵誤解**: 以為 encoder 接受原始波形 (waveform)，實際上需要 **Mel Spectrogram**。

原因分析：
1. `AudioEncoder.conv1` 是 `Conv1d(n_mels=128, d_model=1280, ...)`
2. 期望輸入形狀：`[batch, 128, time]` (128 個 mel bands)
3. 我們傳入的是：`[batch, 1, 240000]` (原始波形)

### 解決方案

#### Step 1: 查看配置
```bash
$ cat models/MiMo-Audio-Tokenizer/config.json | grep -E '"n_mels"|"sampling_rate"|"hop_length"'
"n_mels": 128,
"sampling_rate": 24000,
"hop_length": 240,
```

#### Step 2: 修改資料載入器
```python
class OpticalDataset(Dataset):
    def __init__(self, ...):
        # 新增 Mel Spectrogram 轉換器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=240,
            n_mels=128,
            f_min=0.0,
            f_max=12000.0  # sample_rate // 2
        )
    
    def load_and_preprocess_audio(self, path: str):
        # 載入波形
        waveform, sr = torchaudio.load(path)
        # ... 重新採樣、轉 mono、padding ...
        
        # 轉換為 mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, 128, time]
        return mel_spec.squeeze(0)  # [128, time]
```

#### Step 3: 修改損失函數
```python
def compute_feature_matching_loss(encoder, noisy_mel, clean_mel, device):
    """
    Args:
        noisy_mel: [batch, 128, time] - Mel spectrogram
        clean_mel: [batch, 128, time] - Mel spectrogram
    """
    mel_lens = torch.tensor([noisy_mel.shape[2]] * batch_size, ...)
    
    noisy_features = encoder.get_features(
        input_features=noisy_mel,  # [batch, 128, time]
        output_length=encoder.get_output_length(mel_lens)
    )[0]
```

### 學到的知識
- **MiMo-Audio Encoder 的輸入是 Mel Spectrogram，不是原始波形**
- Conv1d 的第一個參數決定了期望的輸入通道數
- 需要根據配置文件 (`config.json`) 來設定 mel transform 參數
- Mel spectrogram 參數必須與模型訓練時一致：
  - `n_mels=128`
  - `hop_length=240`
  - `sample_rate=24000`

---

## 🐛 問題 4: Device 不匹配

### 錯誤訊息
```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

### 問題原因
LoRA 層在初始化時創建在 CPU 上，而原始模型在 GPU 上。

### 解決方案
```python
def inject_lora_to_encoder(encoder, rank=8, alpha=16.0):
    for name, module in modules_to_modify:
        lora = LoRALayer(...)
        
        # ✅ 將 LoRA 移到與原始模組相同的 device
        lora = lora.to(module.weight.device)
        
        # 註冊參數
        module.lora_A = lora.lora_A
        module.lora_B = lora.lora_B
```

### 學到的知識
- 新創建的 nn.Module 預設在 CPU 上
- 需要確保所有參數在同一個 device 上
- 可以通過 `module.weight.device` 獲取原始模組的 device

---

## 🐛 問題 5: Flash Attention 的 dtype 限制

### 錯誤訊息
```python
RuntimeError: FlashAttention only support fp16 and bf16 data type
```

### 問題原因
MiMo-Audio 使用 Flash Attention，但：
1. 輸入資料（mel spectrogram）是 `float32`
2. Flash Attention 只支援 `float16` 或 `bfloat16`

### 解決方案

#### Step 1: 轉換模型為 bfloat16
```python
tokenizer = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
tokenizer = tokenizer.to(device)
tokenizer = tokenizer.to(torch.bfloat16)  # ✅ 轉換模型
```

#### Step 2: 轉換輸入資料
```python
def compute_feature_matching_loss(encoder, noisy_mel, clean_mel, device):
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        noisy_features = encoder.get_features(
            input_features=noisy_mel.to(torch.bfloat16),  # ✅ 轉換輸入
            output_length=encoder.get_output_length(mel_lens)
        )[0]
```

### 學到的知識
- Flash Attention 對 dtype 有嚴格要求
- `bfloat16` 比 `float16` 更穩定（更大的動態範圍）
- 需要同時轉換模型和輸入資料
- 使用 `torch.cuda.amp.autocast` 時要指定 `dtype=torch.bfloat16`

---

## ✅ 最終成功的配置

### 硬體環境
- GPU: NVIDIA RTX 5090 (24GB)
- CUDA: 12.8.0
- PyTorch: 2.7.0

### 模型配置
```python
# MiMo-Audio-Tokenizer 配置
n_mels: 128
sampling_rate: 24000
hop_length: 240
n_fft: 1024

# LoRA 配置
rank: 8
alpha: 16.0
target_modules: ['attn', 'fc1', 'fc2', 'mlp']
trainable_params: 661M / 1290M (51.22%)
```

### 訓練參數
```bash
python finetune_encoder.py \
    --train-split data/splits/finetune_optical_test/train.json \
    --val-split data/splits/finetune_optical_test/val.json \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --epochs 1 \
    --lora-rank 8 \
    --output-dir ./outputs/finetune_test \
    --num-workers 0
```

### 訓練結果
```
Epoch 1: 100%|██████████| 5/5 [00:00<00:00, 5.89it/s]
Train Loss: 14.2695
Val Loss: 14.9043
Speed: 5.89 it/s (train), 16.47 it/s (val)
```

---

## 🎓 核心學習重點

### 1. MiMo-Audio 的資料流
```
原始音訊 (24kHz waveform)
    ↓
Mel Spectrogram (128 bands, hop=240)
    ↓
AudioEncoder.conv1 (Conv1d: 128→1280)
    ↓
AudioEncoder.conv2 (Conv1d: 1280→1280)
    ↓
Transformer Layers (with LoRA)
    ↓
Features (for loss computation)
```

### 2. LoRA 注入的正確方式
- 先收集所有目標模組（轉為 list）
- 再進行修改（避免迭代時修改）
- 確保 device 一致
- 保持 dtype 兼容性

### 3. Flash Attention 的要求
- 只支援 fp16/bf16
- 需要轉換模型和輸入
- bfloat16 更適合訓練（穩定性）

### 4. 除錯策略
1. **閱讀源碼** - 不要假設 API 行為
2. **檢查形狀** - 使用 `.shape` 和 `print()` 調試
3. **查看配置** - 檢查 `config.json` 確認期望參數
4. **漸進式測試** - 小 batch size 先測試
5. **記錄錯誤** - 完整的 traceback 很重要

---

## 📝 建議的後續工作

### 1. 正式訓練
```bash
# 使用完整資料集訓練
docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace \
    mimo-audio:latest python finetune_encoder.py \
    --train-split data/splits/finetune_optical/train.json \
    --val-split data/splits/finetune_optical/val.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 10 \
    --lora-rank 16 \
    --output-dir ./outputs/optical_lora_r16
```

### 2. 實驗對比
- [ ] 測試不同 LoRA rank (8, 16, 32, 64)
- [ ] 比較 ICL vs Fine-tuning 效果
- [ ] 測試不同的 loss function（MSE vs L1）
- [ ] 嘗試 full fine-tuning vs LoRA

### 3. 文檔完善
- [ ] 更新 `FINETUNE_README.md`
- [ ] 建立 troubleshooting guide
- [ ] 記錄最佳實踐

---

## 🔗 相關文件

- `finetune_encoder.py` - 主要訓練腳本
- `FINETUNE_README.md` - 快速入門指南
- `FINETUNING_GUIDE.md` - 詳細說明
- `FINETUNE_ARCHITECTURE_ANALYSIS.md` - 架構分析
- `src/mimo_audio_tokenizer/modeling_audio_tokenizer.py` - 模型源碼

---

## 💡 給未來自己的建議

1. **永遠先看源碼** - 不要假設 API 的行為
2. **檢查配置文件** - `config.json` 包含關鍵資訊
3. **從簡單開始** - 小資料集 + 小 batch size 先測試
4. **記錄一切** - 包括失敗的嘗試和思考過程
5. **建立測試集** - 用小資料集快速驗證

**最重要的**: 每個錯誤都是學習的機會！

---

**作者**: GitHub Copilot  
**審閱**: Hank  
**更新**: 2025-11-13
