# MiMo-Audio Encoder Fine-tuning 與 Inference 深度分析

**分析日期**: 2025-11-14  
**目的**: 深入理解 MiMo-Audio 如何進行 encoder fine-tuning 和 inference

---

## 📚 目錄

1. [架構總覽](#架構總覽)
2. [Fine-tuning 流程分析](#fine-tuning-流程分析)
3. [Inference 流程分析](#inference-流程分析)
4. [關鍵問題與解答](#關鍵問題與解答)
5. [當前代碼的問題](#當前代碼的問題)
6. [建議修正方案](#建議修正方案)

---

## 架構總覽

### MiMo-Audio 完整架構

```
┌──────────────────────────────────────────────────────────────────┐
│                    MiMo-Audio 完整系統                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1️⃣ MiMo-Audio-Tokenizer (1.2B 參數)                              │
│     ├─ Audio Encoder (我們正在微調的部分)                          │
│     │   ├─ Conv layers (降采樣)                                   │
│     │   ├─ 8 × Transformer layers                                 │
│     │   └─ Residual Vector Quantizer (RVQ)                        │
│     │                                                              │
│     └─ Audio Decoder                                              │
│         ├─ Dequantizer                                            │
│         ├─ 8 × Transformer layers                                 │
│         └─ Vocoder (30 × Transformer layers)                      │
│                                                                    │
│  2️⃣ MiMo-Audio-7B (語言模型)                                       │
│     └─ 基於 Qwen2-7B，處理離散的 audio tokens                      │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 數據流動

```
原始音訊 (Waveform @ 24kHz)
    ↓
Mel Spectrogram [batch, 128, time]
    ↓ 
┌─────────────────────────────────────────┐
│ Audio Encoder                            │
│  ↓ Conv1d (128 → 1280)                  │
│  ↓ Conv1d (stride=2, downsampling)     │
│  ↓ 8 × Transformer layers               │
│  ↓ Hidden States [T', 1280]             │ ← 我們微調這部分
│  ↓ RVQ (8 layers)                       │
│  ↓ Discrete Codes [8, T]                │
└─────────────────────────────────────────┘
    ↓
語言模型處理 / Audio Decoder 重建
```

---

## Fine-tuning 流程分析

### 1. 訓練腳本：`finetune_encoder.py`

#### 核心設計理念

```python
# 目標：讓 encoder 從噪音音訊中提取的特徵接近乾淨音訊的特徵
def compute_feature_matching_loss(encoder, noisy_mel, clean_mel, device):
    """
    Feature Matching Loss:
    - 編碼噪音音訊 → 特徵 A
    - 編碼乾淨音訊 → 特徵 B (frozen, 作為目標)
    - 損失 = MSE(特徵 A, 特徵 B)
    """
    # 噪音音訊的特徵（允許梯度）
    noisy_features = encoder.get_features(
        input_features=noisy_mel,  # [batch, 128, time]
        output_length=encoder.get_output_length(mel_lens)
    )[0]
    
    # 乾淨音訊的特徵（作為目標，不計算梯度）
    with torch.no_grad():
        clean_features = encoder.get_features(
            input_features=clean_mel,
            output_length=encoder.get_output_length(mel_lens)
        )[0]
    
    # MSE 損失
    loss = F.mse_loss(noisy_features, clean_features)
    return loss
```

#### LoRA 注入策略

```python
def inject_lora_to_encoder(encoder, rank=8, alpha=16.0):
    """
    只對 Encoder 的線性層注入 LoRA
    - 目標：attention 和 FFN 的線性層
    - 凍結原始權重
    - 只訓練 LoRA 參數
    """
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear) and \
           any(kw in name for kw in ['attn', 'fc1', 'fc2']):
            # 創建 LoRA 層
            lora = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha
            )
            
            # 凍結原始權重
            module.weight.requires_grad = False
            
            # Monkey-patch forward 方法
            original_forward = module.forward
            def forward_with_lora(x):
                base_output = original_forward(x)
                lora_output = lora(x)
                return base_output + lora_output
            module.forward = forward_with_lora
```

#### 數據處理流程

```python
class OpticalDataset(Dataset):
    def __init__(self, ...):
        # 關鍵：使用與 MiMo-Audio-Tokenizer 一致的配置
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,    # 必須與模型一致
            n_fft=1024,           # 必須與模型一致
            hop_length=240,       # 必須與模型一致
            n_mels=128,           # 必須與模型一致
        )
    
    def load_and_preprocess_audio(self, path):
        # 1. 載入波形
        waveform, sr = torchaudio.load(path)
        
        # 2. 重新採樣到 24kHz
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, sr, 24000)
        
        # 3. 轉換為 mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 4. 截斷或填充到固定長度
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            waveform = F.pad(waveform, (0, pad_length))
        
        # 5. 轉換為 Mel Spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, 128, time]
        
        return mel_spec.squeeze(0)  # [128, time]
```

### 2. 訓練循環

```python
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    for batch in dataloader:
        noisy_audio = batch['noisy_audio'].to(device)  # [B, 128, T]
        clean_audio = batch['clean_audio'].to(device)  # [B, 128, T]
        
        # 計算特徵匹配損失
        loss = compute_feature_matching_loss(
            model.encoder, 
            noisy_audio, 
            clean_audio, 
            device
        )
        
        # 梯度累積 + 優化
        loss.backward()
        optimizer.step()
```

### 3. Checkpoint 保存

```python
def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    # 只保存 LoRA 參數
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[name] = param.cpu()
    
    checkpoint = {
        'epoch': epoch,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

---

## Inference 流程分析

### 1. 模型載入：`test_finetuned_inference.py`

```python
def load_finetuned_model(checkpoint_path, tokenizer_path, device='cuda'):
    """
    載入微調後的模型的步驟：
    1. 載入基礎 tokenizer
    2. 重新注入 LoRA (相同配置)
    3. 載入 LoRA 權重
    """
    # Step 1: 載入基礎 tokenizer
    tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    tokenizer = tokenizer.to(device).to(torch.bfloat16)
    
    # Step 2: 重新注入 LoRA (必須與訓練時一致)
    inject_lora_to_encoder(
        tokenizer.encoder,
        rank=32,
        alpha=64
    )
    
    # Step 3: 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer.encoder.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    
    return tokenizer
```

### 2. 音訊增強流程

```python
def enhance_audio(model, input_audio_path, output_audio_path, device):
    # Step 1: 載入音訊
    waveform, sr = torchaudio.load(input_audio_path)
    waveform = preprocess_waveform(waveform, sr)  # 重新採樣、轉 mono
    
    # Step 2: 轉換為 Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=128
    ).to(device)
    mel_spec = mel_transform(waveform)  # [1, 128, T]
    
    # Step 3: 編碼（使用微調後的 encoder）
    with torch.no_grad():
        input_lens = torch.tensor([mel_spec.shape[2]], device=device)
        
        # 🔴 問題：這裡的 API 調用方式
        hidden_states, _, output_length, codes = model.encoder.encode(
            input_features=mel_spec,
            input_lens=input_lens,
            use_quantizer=True
        )
    
    # Step 4: 解碼回音訊
    with torch.no_grad():
        reconstructed_audio = model.decode(codes)
    
    # Step 5: 保存
    torchaudio.save(output_audio_path, reconstructed_audio, 24000)
```

### 3. MimoAudio 類中的 Inference

在 `src/mimo_audio/mimo_audio.py` 中：

```python
class MimoAudio:
    def encode_batch(self, input_features, input_lens, max_length=256000):
        """
        批次編碼音訊特徵
        """
        feature_groups, len_groups = self.group_by_length(
            input_features, input_lens, max_length
        )
        
        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features.to(self.device),
                    input_lens=lengths.to(self.device),
                    return_codes_only=True
                )
                encoded_parts.append(codes)
        
        return torch.cat(encoded_parts, dim=-1)
    
    def preprocess_input(self, input):
        """
        預處理輸入音訊
        """
        # 1. 載入音訊
        if isinstance(input, str):
            wav, sr = torchaudio.load(input)
        else:
            wav = input
        
        # 2. 重新採樣
        wav = self.resample_audio_if_needed(wav, sr)
        
        # 3. 轉換為 Mel
        mel = self.wav2mel(wav).transpose(0, 1)  # (T, n_mels)
        
        # 4. 分段處理（避免記憶體溢出）
        input_len = mel.size(0)
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)
        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)
        
        # 5. 批次編碼
        codes_packed = self.encode_batch(
            input_features=mel,
            input_lens=torch.tensor(input_len_seg),
        )
        
        return codes_packed
```

---

## 關鍵問題與解答

### Q1: 微調會動到原先小米模型的 token 嗎？

**答案：不會！**

原因：
1. **LoRA 不改變原始權重**：原始的 encoder 權重被凍結（`requires_grad=False`）
2. **Token 由 RVQ 決定**：discrete tokens 是由 Residual Vector Quantizer (RVQ) 產生的
3. **RVQ 的 codebook 不變**：我們沒有微調 quantizer，只微調 encoder 的特徵提取部分

流程：
```
Noisy Audio → Encoder (微調) → Hidden States → RVQ (凍結) → Tokens (不變)
                  ↑                                ↑
            只調整這部分                    codebook 保持一致
```

### Q2: 我們是透過調整 encoder 讓新的特徵能對應到原先的 token 空間嗎？

**答案：是的！**

具體來說：
1. **目標**：讓噪音音訊經過 encoder 後的 hidden states 接近乾淨音訊的 hidden states
2. **方法**：Feature Matching Loss = MSE(noisy_features, clean_features)
3. **效果**：調整後的 encoder 可以從噪音中提取更乾淨的特徵
4. **保持相容**：這些特徵仍然在原始的 token 空間中（因為 RVQ 沒變）

```
原始流程：
Noisy Audio → Encoder → Noisy Features → RVQ → Noisy Tokens

微調後：
Noisy Audio → Encoder (微調) → Clean-like Features → RVQ (same) → Clean-like Tokens
                                        ↑
                              更接近 clean audio 的特徵
                              但仍然在相同的量化空間
```

### Q3: 中間需要轉換器嗎？

**答案：不需要！**

原因：
1. **RVQ 作為橋樑**：RVQ 的 codebook 是固定的，它自動處理連續特徵到離散 token 的映射
2. **特徵空間一致**：微調只改善特徵質量，不改變特徵空間的維度和結構
3. **端到端相容**：微調後的 encoder 輸出可以直接送入 decoder 或語言模型

### Q4: 訓練時使用的是什麼數據格式？

**答案：Mel Spectrogram**

關鍵配置：
```python
# 必須與 MiMo-Audio-Tokenizer 的配置一致
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000,      # config.json 中的 sampling_rate
    n_fft=1024,             # config.json 中的 nfft
    hop_length=240,         # config.json 中的 hop_length
    n_mels=128,             # 這是特定的，但需要與 encoder 期望一致
)
```

檢查方式：
```bash
# 查看 config.json
cat models/MiMo-Audio-Tokenizer/config.json | grep -E '"n_mels"|"sampling_rate"|"hop_length"|"nfft"'

# 查看 encoder 的第一層
# Conv1d(in_channels=128, ...) 表示期望 128 個 mel bands
```

### Q5: `input_lens` 應該是什麼？

**這是當前代碼的核心問題！**

根據源碼分析：

```python
# 在 src/mimo_audio_tokenizer/modeling_audio_tokenizer.py 中：

class AudioEncoder:
    def get_output_length(self, mel_len):
        """
        計算輸出長度
        Args:
            mel_len: Mel spectrogram 的時間維度長度
        """
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1
    
    def encode(self, input_features, input_lens=None, ...):
        """
        Args:
            input_features: [batch, n_mels, time] - Mel spectrogram
            input_lens: Mel spectrogram 的時間維度長度（每個樣本的）
        """
        if output_length is None:
            output_length = self.get_output_length(input_lens)
```

結論：
- **`input_lens` 應該是 Mel Spectrogram 的時間維度**
- 不是原始波形的長度！
- 當前訓練代碼是正確的：`input_lens = torch.tensor([mel_spec.shape[2]])`

但要注意：
```python
# MiMoAudioTokenizer.get_output_length 的定義
def get_output_length(self, mel_len):
    """
    Args:
        mel_len: Mel spectrogram 的 時間步數
    """
    tgt_len = mel_len + 3 - self.config.kernel_size
    return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1
```

---

## 當前代碼的問題

### 問題 1: Inference 測試失敗

根據 `INFERENCE_TEST_RESULT.md`：

```
錯誤: RuntimeError: CUDA error: device-side assert triggered
位置: unpack_hidden_states() 函數
```

**但經過深入分析，這可能不是 `input_lens` 的問題**。

讓我重新檢查 `test_finetuned_inference.py`：

```python
# 當前代碼
def enhance_audio(model, input_audio_path, ...):
    mel_spec = mel_transform(waveform)  # [1, 128, T]
    
    with torch.no_grad():
        # ❓ 這裡使用的 API
        input_lens = torch.tensor([mel_spec.shape[2]], device=device)
        encoded = model.encode(mel_spec, input_lens=input_lens)
```

問題：
1. `model` 是 `MiMoAudioTokenizer`
2. `MiMoAudioTokenizer.encode()` 返回 4 個值：
   ```python
   def encode(self, mels, input_lens, use_quantizer=True):
       return hidden_states, hidden_states_packed, encoder_output_length, codes
   ```
3. 但測試代碼期望不同的返回值

### 問題 2: LoRA 權重載入

```python
def load_finetuned_model(...):
    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ❌ 問題：直接 load_state_dict
    tokenizer.encoder.load_state_dict(checkpoint['lora_state_dict'], strict=False)
```

問題：
1. Checkpoint 中保存的是完整的參數名稱（包括 `encoder.layers.0.self_attn.q_proj.lora_A`）
2. 但這些參數已經通過 monkey-patch 註冊到 module 上
3. 需要確保載入方式正確

### 問題 3: Decode 方法使用錯誤

```python
# test_finetuned_inference.py 中
def enhance_audio(...):
    # 編碼
    encoded = model.encode(mel_spec, input_lens=input_lens)
    
    # ❌ 解碼
    decoded_mel = model.decode(encoded, input_lens=input_lens)
```

問題：
1. `MiMoAudioTokenizer.decode()` 接受 **codes** (離散 token)，不是 hidden states
2. 正確的簽名：
   ```python
   def decode(self, codes):
       """
       Args:
           codes: [num_quantizers, time] - 離散的 token IDs
       Returns:
           waveform: [batch, 1, samples]
       """
   ```

---

## 建議修正方案

### 修正 1: 更新 Inference 測試腳本

```python
# test_finetuned_inference.py

def enhance_audio(model, input_audio_path, output_audio_path, device='cuda'):
    """使用微調後的 encoder 增強音訊"""
    
    # 1. 載入音訊
    waveform, sr = torchaudio.load(input_audio_path)
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 2. 轉換為 Mel
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=128
    ).to(device)
    
    mel_spec = mel_transform(waveform.to(device))  # [1, 128, T]
    
    # 3. 編碼（使用微調後的 encoder）
    with torch.no_grad():
        # input_lens 是 mel 的時間維度
        input_lens = torch.tensor([mel_spec.shape[2]], device=device)
        
        # ✅ 正確的 API 調用
        hidden_states, hidden_states_packed, output_length, codes = model.encode(
            mels=mel_spec,
            input_lens=input_lens,
            use_quantizer=True
        )
        
        print(f"Encoded codes shape: {codes.shape}")  # [8, T']
        
        # 4. 解碼回波形
        reconstructed_audio = model.decode(codes)  # [1, 1, samples]
    
    # 5. 保存
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(
        output_audio_path,
        reconstructed_audio.squeeze(0).cpu(),
        24000
    )
    
    print(f"✅ 增強完成：{output_audio_path}")
```

### 修正 2: 更新 LoRA 載入邏輯

```python
def load_finetuned_model(checkpoint_path, tokenizer_path, device='cuda'):
    """載入微調後的模型"""
    
    # 1. 載入基礎 tokenizer
    tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    tokenizer = tokenizer.to(device).to(torch.bfloat16)
    
    # 2. 重新注入 LoRA（與訓練時相同的配置）
    lora_modules = inject_lora_to_encoder(
        tokenizer.encoder,
        rank=32,
        alpha=64
    )
    
    # 3. 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 4. 手動載入 LoRA 權重
    lora_state = checkpoint['lora_state_dict']
    
    # 方式 1: 直接設置參數值
    for name, param in tokenizer.encoder.named_parameters():
        if name in lora_state:
            param.data = lora_state[name].to(device).to(torch.bfloat16)
            print(f"✓ Loaded: {name}")
    
    # 或方式 2: 使用 load_state_dict（移除 'encoder.' 前綴）
    # encoder_state = {k.replace('encoder.', ''): v for k, v in lora_state.items()}
    # tokenizer.encoder.load_state_dict(encoder_state, strict=False)
    
    print(f"✅ 載入了 {len(lora_state)} 個 LoRA 參數")
    
    return tokenizer
```

### 修正 3: 驗證訓練代碼的正確性

訓練代碼 `finetune_encoder.py` 大部分是正確的，但需要確認：

```python
def compute_feature_matching_loss(encoder, noisy_mel, clean_mel, device):
    """計算特徵匹配損失"""
    batch_size = noisy_mel.shape[0]
    mel_len = noisy_mel.shape[2]  # Mel 的時間維度
    
    # ✅ 正確：input_lens 是 mel 的時間維度
    mel_lens = torch.tensor([mel_len] * batch_size, device=device, dtype=torch.long)
    
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # 編碼噪音音訊
        noisy_features = encoder.get_features(
            input_features=noisy_mel.to(torch.bfloat16),
            output_length=encoder.get_output_length(mel_lens)
        )[0]
        
        # 編碼乾淨音訊（作為目標）
        with torch.no_grad():
            clean_features = encoder.get_features(
                input_features=clean_mel.to(torch.bfloat16),
                output_length=encoder.get_output_length(mel_lens)
            )[0]
        
        loss = F.mse_loss(noisy_features, clean_features)
    
    return loss
```

這部分是正確的！

### 修正 4: 整合到 MimoAudio 類

如果要將微調後的 encoder 整合到完整的 MiMo-Audio 系統：

```python
class MimoAudio:
    def __init__(self, model_path, mimo_audio_tokenizer_path, 
                 finetuned_encoder_checkpoint=None):
        # 原始初始化
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
            mimo_audio_tokenizer_path
        )
        
        # 如果有微調的 encoder，載入它
        if finetuned_encoder_checkpoint:
            self.load_finetuned_encoder(finetuned_encoder_checkpoint)
    
    def load_finetuned_encoder(self, checkpoint_path):
        """載入微調後的 encoder"""
        print(f"Loading finetuned encoder from {checkpoint_path}")
        
        # 注入 LoRA
        from finetune_encoder import inject_lora_to_encoder
        inject_lora_to_encoder(self.mimo_audio_tokenizer.encoder, rank=32, alpha=64)
        
        # 載入權重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        for name, param in self.mimo_audio_tokenizer.encoder.named_parameters():
            if name in checkpoint['lora_state_dict']:
                param.data = checkpoint['lora_state_dict'][name].to(self.device)
        
        print("✅ Finetuned encoder loaded!")
```

---

## 總結與建議

### ✅ 正確的理解

1. **Fine-tuning 方法**：
   - 使用 LoRA 微調 encoder 的線性層
   - 通過 Feature Matching Loss 讓噪音特徵接近乾淨特徵
   - RVQ (quantizer) 保持凍結，確保 token 空間一致

2. **數據格式**：
   - 輸入：Mel Spectrogram `[batch, 128, time]`
   - `input_lens` 是 mel 的時間維度
   - 必須與 tokenizer 配置一致

3. **架構設計**：
   - Encoder 提取特徵（我們微調這部分）
   - RVQ 量化特徵（保持不變）
   - Decoder 重建音訊（保持不變）

### 🔧 需要修正的問題

1. **Inference 腳本**：
   - API 調用方式需要修正
   - 返回值處理需要修正
   - Decode 方法使用需要修正

2. **LoRA 權重載入**：
   - 需要確保參數名稱匹配
   - 需要正確處理 device 和 dtype

3. **測試驗證**：
   - 需要完整的端到端測試
   - 需要驗證音質改善效果

### 📋 下一步行動

1. **立即修正**：
   - 更新 `test_finetuned_inference.py` 使用正確的 API
   - 測試微調後的模型是否能正常 encode-decode
   - 驗證輸出音質

2. **後續改進**：
   - 實驗不同的 LoRA rank (8, 16, 32, 64)
   - 比較 Feature Matching Loss vs MSE on Codes
   - 整合到完整的 MiMo-Audio pipeline

3. **文檔完善**：
   - 更新 README 說明正確的使用方式
   - 建立 troubleshooting guide
   - 記錄最佳實踐和參數設置

---

**最重要的發現**：
- 訓練代碼基本正確 ✅
- Inference 代碼需要修正 ❌
- 架構理解已經清晰 ✅
- 下一步應該專注於修正 inference 並測試效果 🎯

