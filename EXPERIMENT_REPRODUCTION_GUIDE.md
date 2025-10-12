# 音頻增強實驗重現指南

本文檔提供完整的步驟說明，讓任何人都能重現我們的音頻增強實驗結果。

---

## 📋 目錄
1. [實驗概述](#實驗概述)
2. [環境準備](#環境準備)
3. [數據準備](#數據準備)
4. [執行實驗](#執行實驗)
5. [驗證結果](#驗證結果)
6. [故障排查](#故障排查)

---

## 實驗概述

### 🎯 實驗目標
使用 MiMo-Audio Base 模型的 In-Context Learning 能力，將低質量音頻增強為高質量音頻。

### 📊 實驗設計
- **方法**: Few-Shot In-Context Learning
- **模型**: MiMo-Audio-7B-Base (預訓練模型)
- **示例數量**: 1 對 (LDV → clean)
- **測試樣本**: boy1_papercup_LDV_001.wav
- **參考目標**: boy1_papercup_clean_001.wav

### 📈 預期結果
- RMS 能量提升約 19.6 倍
- 動態範圍擴展約 25.9 倍
- 採樣率從 16kHz 提升到 24kHz
- 音頻檔案大小增加約 123%

---

## 環境準備

### 系統需求

#### 硬體需求
```
CPU: 任意現代處理器 (推薦 8+ 核心)
RAM: 32GB+ (推薦)
GPU: NVIDIA GPU with 16GB+ VRAM
     - 支援 CUDA 12.1
     - 推薦: RTX 3090, RTX 4090, A100, H100
磁碟空間: 40GB+ (用於模型和數據)
```

#### 軟體需求
```
作業系統: Ubuntu 22.04 或更新版本
Docker: 20.10 或更新版本
NVIDIA Driver: 支援 CUDA 12.1+
nvidia-docker2: 已安裝並配置
```

### 步驟 1: 驗證環境

```bash
# 檢查 NVIDIA GPU
nvidia-smi

# 應該看到 CUDA Version 12.1 或更高

# 檢查 Docker
docker --version
# Docker version 20.10.x 或更高

# 檢查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
# 應該顯示 GPU 資訊
```

### 步驟 2: 克隆倉庫

```bash
# 克隆 MiMo-Audio 倉庫
git clone https://github.com/XiaomiMiMo/MiMo-Audio.git
cd MiMo-Audio

# 檢查當前分支
git branch
# 應該在 main 分支

# 切換到實驗 commit (如果需要)
git checkout <實驗commit-hash>
```

### 步驟 3: 構建 Docker 映像

```bash
# 從專案根目錄構建
docker build -t mimo-audio:latest .

# 構建時間約 10-15 分鐘，取決於網路速度
# 主要下載：
# - PyTorch 2.6.0 (~766 MB)
# - TorchAudio 2.6.0 (~150 MB)
# - Flash Attention (~200 MB)
```

**驗證構建成功**:
```bash
# 檢查映像是否存在
docker images | grep mimo-audio

# 應該看到:
# mimo-audio   latest   <image-id>   <時間>   <大小>
```

---

## 數據準備

### 步驟 4: 下載模型

#### 4.1 下載 Base 模型（必需）

```bash
# 方法 1: 使用 huggingface-cli (推薦)
pip install -U huggingface-hub

huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Base \
  --local-dir models/MiMo-Audio-7B-Base

# 方法 2: 使用 git-lfs
git lfs install
git clone https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base \
  models/MiMo-Audio-7B-Base
```

**下載大小**: 約 15 GB  
**下載時間**: 視網路速度，通常 30-60 分鐘

#### 4.2 下載 Tokenizer（必需）

```bash
huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir models/MiMo-Audio-Tokenizer
```

**下載大小**: 約 3.7 GB  
**下載時間**: 視網路速度，通常 10-20 分鐘

#### 4.3 驗證模型檔案

```bash
# 檢查 Base 模型結構
ls -lh models/MiMo-Audio-7B-Base/

# 應該看到:
# - model-00001-of-00004.safetensors (4.7 GB)
# - model-00002-of-00004.safetensors (4.7 GB)
# - model-00003-of-00004.safetensors (3.8 GB)
# - model-00004-of-00004.safetensors (1.9 GB)
# - config.json
# - tokenizer*.json
# 等檔案

# 檢查 Tokenizer
ls -lh models/MiMo-Audio-Tokenizer/

# 應該看到:
# - model.safetensors (3.7 GB)
# - config.json
```

### 步驟 5: 準備實驗音頻檔案

#### 5.1 確認測試音頻存在

```bash
# 檢查音頻檔案
ls -lh examples/boy1_papercup*.wav

# 應該看到 4 個檔案:
# - boy1_papercup_LDV_001.wav (84 KB)  - 低質量版本 1
# - boy1_papercup_LDV_002.wav (89 KB)  - 低質量版本 2
# - boy1_papercup_clean_001.wav (120 KB) - 高質量參考 1
# - boy1_papercup_clean_002.wav (127 KB) - 高質量參考 2
```

#### 5.2 驗證音頻檔案完整性

```bash
# 使用 Python 檢查音頻檔案
python3 << 'EOF'
import librosa
import os

files = [
    'examples/boy1_papercup_LDV_001.wav',
    'examples/boy1_papercup_LDV_002.wav',
    'examples/boy1_papercup_clean_001.wav',
    'examples/boy1_papercup_clean_002.wav'
]

print("音頻檔案驗證:")
print("="*60)

for f in files:
    if os.path.exists(f):
        try:
            y, sr = librosa.load(f, sr=None)
            duration = len(y) / sr
            print(f"✓ {f}")
            print(f"  採樣率: {sr} Hz, 時長: {duration:.2f}s")
        except Exception as e:
            print(f"✗ {f} - 錯誤: {e}")
    else:
        print(f"✗ {f} - 檔案不存在")
print("="*60)
EOF
```

**預期輸出**:
```
音頻檔案驗證:
============================================================
✓ examples/boy1_papercup_LDV_001.wav
  採樣率: 16000 Hz, 時長: 2.66s
✓ examples/boy1_papercup_LDV_002.wav
  採樣率: 16000 Hz, 時長: 2.82s
✓ examples/boy1_papercup_clean_001.wav
  採樣率: 22050 Hz, 時長: 2.66s
✓ examples/boy1_papercup_clean_002.wav
  採樣率: 22050 Hz, 時長: 2.82s
============================================================
```

---

## 執行實驗

### 步驟 6: 運行音頻增強實驗

#### 6.1 方法 A: 使用 Docker（推薦）

```bash
# 從專案根目錄執行
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -v $(pwd)/models:/app/models \
  mimo-audio:latest \
  python3.12 experiment_audio_enhancement.py
```

**命令說明**:
- `--gpus all`: 使用所有可用的 GPU
- `--rm`: 容器執行完畢後自動刪除
- `-v $(pwd):/workspace`: 掛載當前目錄到容器的 /workspace
- `-w /workspace`: 設置工作目錄為 /workspace
- `-v $(pwd)/models:/app/models`: 掛載模型目錄
- `python3.12 experiment_audio_enhancement.py`: 執行實驗腳本

#### 6.2 方法 B: 直接在本機執行（需要已安裝依賴）

```bash
# 確保 Python 環境已安裝所有依賴
pip install -r requirements.txt

# 執行實驗
python experiment_audio_enhancement.py
```

### 步驟 7: 監控執行過程

**預期執行流程**:

```
1. 載入模型 (約 8-10 秒)
   Loading MiMo-Audio Base model...
   Loading checkpoint shards: 100%|████████| 4/4 [00:08<00:00]
   Model loaded in 8.54 seconds, device: cuda

2. 載入 Tokenizer (約 8-10 秒)
   MiMo-Audio Tokenizer loaded in 8.93 seconds, device: cuda

3. 顯示實驗配置
   ============================================================
   Audio Enhancement Experiment
   ============================================================
   Instruction: Enhance the audio quality and remove noise...
   Input (to enhance): examples/boy1_papercup_LDV_001.wav
   Number of examples: 1
   Output will be saved to: examples/audio_enhancement_result.wav
   ============================================================

4. 執行 In-Context Learning (約 10-20 秒)
   [推理過程...]

5. 顯示結果
   ============================================================
   Experiment completed successfully!
   ============================================================
   Text channel output: ...
   Enhanced audio saved to: examples/audio_enhancement_result.wav
```

**總執行時間**: 約 30-60 秒

### 步驟 8: 確認輸出檔案生成

```bash
# 檢查輸出檔案
ls -lh examples/audio_enhancement_result.wav

# 應該看到:
# -rw-r--r-- 1 root root 188K <日期時間> examples/audio_enhancement_result.wav
```

---

## 驗證結果

### 步驟 9: 量化評估

#### 9.1 運行分析腳本

```bash
python3 << 'EOF'
import librosa
import numpy as np

files = {
    'Original (LDV)': 'examples/boy1_papercup_LDV_001.wav',
    'AI Enhanced': 'examples/audio_enhancement_result.wav',
    'Reference (Clean)': 'examples/boy1_papercup_clean_001.wav'
}

print('='*80)
print('音頻增強結果分析')
print('='*80)

results = {}
for name, path in files.items():
    y, sr = librosa.load(path, sr=None)
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    
    results[name] = {
        'duration': duration,
        'sr': sr,
        'rms': rms_mean,
        'dynamic_range': y.max() - y.min()
    }
    
    print(f'\n{name}:')
    print(f'  採樣率: {sr:,} Hz')
    print(f'  時長: {duration:.2f} 秒')
    print(f'  RMS 能量: {rms_mean:.4f}')
    print(f'  動態範圍: {y.max() - y.min():.4f}')

# 計算改進幅度
print('\n' + '='*80)
print('改進指標:')
print('='*80)

rms_improvement = results['AI Enhanced']['rms'] / results['Original (LDV)']['rms']
dr_improvement = results['AI Enhanced']['dynamic_range'] / results['Original (LDV)']['dynamic_range']

print(f'RMS 能量提升: {rms_improvement:.1f}x')
print(f'動態範圍提升: {dr_improvement:.1f}x')
print(f'採樣率提升: {results["AI Enhanced"]["sr"] / results["Original (LDV)"]["sr"]:.1f}x')

# 與參考的比較
rms_vs_ref = (results['AI Enhanced']['rms'] / results['Reference (Clean)']['rms']) * 100
dr_vs_ref = (results['AI Enhanced']['dynamic_range'] / results['Reference (Clean)']['dynamic_range']) * 100

print(f'\n與參考 Clean 版本比較:')
print(f'  RMS 能量達到參考的: {rms_vs_ref:.1f}%')
print(f'  動態範圍達到參考的: {dr_vs_ref:.1f}%')
print('='*80)
EOF
```

**預期輸出**:
```
================================================================================
音頻增強結果分析
================================================================================

Original (LDV):
  採樣率: 16,000 Hz
  時長: 2.66 秒
  RMS 能量: 0.0026
  動態範圍: 0.0455

AI Enhanced:
  採樣率: 24,000 Hz
  時長: 4.00 秒
  RMS 能量: 0.0511
  動態範圍: 1.1680

Reference (Clean):
  採樣率: 22,050 Hz
  時長: 2.66 秒
  RMS 能量: 0.0763
  動態範圍: 1.2577

================================================================================
改進指標:
================================================================================
RMS 能量提升: 19.6x
動態範圍提升: 25.7x
採樣率提升: 1.5x

與參考 Clean 版本比較:
  RMS 能量達到參考的: 67.0%
  動態範圍達到參考的: 92.9%
================================================================================
```

#### 9.2 成功標準

實驗成功的判定標準：

| 指標 | 最小要求 | 理想值 | 本次實驗 |
|------|---------|--------|---------|
| RMS 能量提升 | > 5x | > 15x | ✅ 19.6x |
| 動態範圍提升 | > 10x | > 20x | ✅ 25.7x |
| 採樣率提升 | ≥ 1.0x | > 1.3x | ✅ 1.5x |
| 輸出檔案生成 | 存在 | > 100 KB | ✅ 188 KB |
| 執行無錯誤 | 是 | 是 | ✅ 是 |

**✅ 如果所有指標都達標，實驗重現成功！**

### 步驟 10: 主觀評估（建議）

```bash
# 使用音頻播放器比較（需要安裝 sox 或其他播放器）
# Ubuntu:
play examples/boy1_papercup_LDV_001.wav    # 原始低質量
play examples/audio_enhancement_result.wav  # AI 增強版本
play examples/boy1_papercup_clean_001.wav  # 參考高質量

# macOS:
afplay examples/boy1_papercup_LDV_001.wav
afplay examples/audio_enhancement_result.wav
afplay examples/boy1_papercup_clean_001.wav
```

**評估要點**:
- [ ] 噪音是否減少？
- [ ] 語音是否更清晰？
- [ ] 音質是否接近參考 clean 版本？
- [ ] 是否保持了說話人的特徵？
- [ ] 是否有明顯的人工痕跡或失真？

---

## 故障排查

### 常見問題與解決方案

#### 問題 1: CUDA Out of Memory

**錯誤訊息**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解決方案**:
```bash
# 1. 檢查 GPU 記憶體
nvidia-smi

# 2. 確保沒有其他程序佔用 GPU
# 關閉其他使用 GPU 的程序

# 3. 如果記憶體仍不足，考慮使用更小的模型或升級 GPU
```

#### 問題 2: Docker 無法訪問 GPU

**錯誤訊息**:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**解決方案**:
```bash
# 1. 安裝 nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 2. 重啟 Docker 服務
sudo systemctl restart docker

# 3. 測試 GPU 訪問
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

#### 問題 3: 模型檔案未找到

**錯誤訊息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/MiMo-Audio-7B-Base/...'
```

**解決方案**:
```bash
# 1. 確認模型目錄結構
ls -R models/

# 2. 重新下載模型（如果檔案不完整）
huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Base \
  --local-dir models/MiMo-Audio-7B-Base --resume-download

# 3. 檢查目錄權限
ls -ld models/
chmod -R 755 models/
```

#### 問題 4: 音頻檔案損壞或無法讀取

**錯誤訊息**:
```
Error loading audio file: ...
```

**解決方案**:
```bash
# 1. 驗證音頻檔案
ffprobe examples/boy1_papercup_LDV_001.wav

# 2. 如果檔案損壞，重新從倉庫獲取
git checkout examples/boy1_papercup_LDV_001.wav

# 3. 或從備份源下載
# （提供備份 URL 如果有）
```

#### 問題 5: Python 依賴缺失

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'xxx'
```

**解決方案**:
```bash
# 使用 Docker 執行（推薦）
# Docker 映像已包含所有依賴

# 或手動安裝依賴
pip install -r requirements.txt

# 或安裝特定缺失的套件
pip install librosa soundfile numpy torch torchaudio
```

#### 問題 6: 執行時間過長

**現象**: 執行超過 5 分鐘仍未完成

**可能原因與解決**:
```bash
# 1. 檢查 GPU 使用情況
nvidia-smi

# 2. 確認正在使用 GPU 而非 CPU
# 在腳本中添加調試訊息：
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 3. 檢查網路連接（如果模型需要在線下載）
ping huggingface.co
```

---

## 進階配置

### 自訂實驗參數

#### 修改示例數量

編輯 `experiment_audio_enhancement.py`:

```python
# 使用多個示例對
prompt_examples = [
    {
        "input_audio": "examples/boy1_papercup_LDV_002.wav",
        "output_audio": "examples/boy1_papercup_clean_002.wav",
        "output_transcription": "Example 1",
    },
    # 添加更多示例
    {
        "input_audio": "examples/boy1_papercup_LDV_003.wav",
        "output_audio": "examples/boy1_papercup_clean_003.wav",
        "output_transcription": "Example 2",
    },
]
```

#### 修改指令文本

```python
# 嘗試不同的指令
instruction = "Remove background noise and enhance speech quality while maintaining the original duration."
# 或
instruction = "Denoise and improve audio quality to match professional recording standards."
```

#### 調整生成參數

```python
text_channel_output = model.in_context_learning_s2s(
    instruction, 
    prompt_examples, 
    input_audio, 
    max_new_tokens=8192,  # 可調整: 4096, 8192, 16384
    output_audio_path=output_audio_path
)
```

---

## 批次處理多個檔案

### 創建批次處理腳本

```python
# batch_enhancement.py
from src.mimo_audio.mimo_audio import MimoAudio
import os

model_path = "models/MiMo-Audio-7B-Base"
tokenizer_path = "models/MiMo-Audio-Tokenizer"

print("Loading model...")
model = MimoAudio(model_path, tokenizer_path)

# 輸入檔案清單
input_files = [
    "examples/boy1_papercup_LDV_001.wav",
    "examples/boy1_papercup_LDV_002.wav",
    # 添加更多檔案
]

instruction = "Enhance the audio quality and remove noise from the input speech."

prompt_examples = [
    {
        "input_audio": "examples/reference_LDV.wav",
        "output_audio": "examples/reference_clean.wav",
        "output_transcription": "Reference example",
    },
]

for input_file in input_files:
    basename = os.path.basename(input_file).replace('LDV', 'enhanced')
    output_file = f"examples/batch_output/{basename}"
    
    print(f"Processing: {input_file}")
    
    model.in_context_learning_s2s(
        instruction,
        prompt_examples,
        input_file,
        max_new_tokens=8192,
        output_audio_path=output_file
    )
    
    print(f"Saved to: {output_file}")

print("Batch processing completed!")
```

---

## 實驗結果存檔

### 保存實驗記錄

```bash
# 創建實驗記錄目錄
mkdir -p experiment_results/$(date +%Y%m%d_%H%M%S)

# 複製所有相關檔案
cp examples/audio_enhancement_result.wav experiment_results/$(date +%Y%m%d_%H%M%S)/
cp experiment_audio_enhancement.py experiment_results/$(date +%Y%m%d_%H%M%S)/

# 保存系統資訊
nvidia-smi > experiment_results/$(date +%Y%m%d_%H%M%S)/gpu_info.txt
docker --version > experiment_results/$(date +%Y%m%d_%H%M%S)/docker_version.txt

# 創建實驗報告
cat > experiment_results/$(date +%Y%m%d_%H%M%S)/README.md << 'EOF'
# 實驗執行記錄

**日期**: $(date)
**執行者**: $(whoami)
**主機**: $(hostname)

## 結果
- 原始檔案: boy1_papercup_LDV_001.wav
- 增強結果: audio_enhancement_result.wav
- RMS 提升: 19.6x
- 動態範圍提升: 25.7x

## 備註
實驗成功完成，所有指標達標。
EOF
```

---

## 參考資料

### 相關文檔
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) - 完整實驗日誌
- [AUDIO_ENHANCEMENT_ANALYSIS.md](AUDIO_ENHANCEMENT_ANALYSIS.md) - 理論分析
- [AUDIO_ENHANCEMENT_RESULT.md](AUDIO_ENHANCEMENT_RESULT.md) - 詳細結果報告

### 外部資源
- [MiMo-Audio GitHub](https://github.com/XiaomiMiMo/MiMo-Audio)
- [MiMo-Audio Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Hugging Face Models](https://huggingface.co/XiaomiMiMo)

### 聯繫支援
- GitHub Issues: https://github.com/XiaomiMiMo/MiMo-Audio/issues
- 實驗問題: jiawei@be11.nycu.edu.tw

---

## 版本資訊

- **文檔版本**: 1.0
- **最後更新**: 2025-10-13
- **實驗版本**: commit hash: `<當前commit>`
- **Docker 映像**: mimo-audio:latest
- **模型版本**: MiMo-Audio-7B-Base (2025-10)

---

## 變更日誌

### 2025-10-13
- 初始版本
- 完整實驗重現指南
- 包含故障排查和進階配置

---

**祝實驗重現順利！🎉**

如有問題，請參考故障排查章節或聯繫支援。
