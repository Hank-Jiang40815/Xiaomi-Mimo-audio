# 環境清理與 Docker 化指南

## 📋 背景

為了保持 host 環境乾淨並實現完全的環境隔離，我們將所有工具都 Docker 化。

## 🎯 目標架構

```
Host 環境
├─ ✅ Docker Engine
├─ ✅ Git
└─ ❌ 無任何 Python ML 套件

Docker 容器
├─ 🎤 whisper-asr:latest     → Whisper ASR 轉錄工具
└─ 🎵 mimo-audio:latest      → MiMo-Audio 模型環境
```

## 📝 清理步驟記錄

### 1. 備份 (已完成 ✅)
```bash
conda list > data/backup_conda_base_env_20251113.txt
pip list | grep -E "whisper|torch|triton|nvidia" > data/whisper_packages_to_remove.txt
```

### 2. 建立 Whisper Docker (進行中 🔄)
- ✅ 創建 `Dockerfile.whisper`
- 🔄 構建映像: `docker build -f Dockerfile.whisper -t whisper-asr:latest .`
- ⏳ 創建便捷腳本: `scripts/run_whisper_docker.sh`

### 3. 清理 Host 環境 (待執行 ⏳)
```bash
# 移除 Whisper 相關套件
pip uninstall -y openai-whisper
pip uninstall -y torch torchaudio
pip uninstall -y triton
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12
pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
pip uninstall -y nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12
pip uninstall -y nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12
pip uninstall -y nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12
```

### 4. 驗證環境 (待執行 ⏳)
```bash
# 檢查 Host 環境
pip list | grep -E "torch|whisper|nvidia"  # 應該沒有結果

# 測試 Whisper Docker
docker run --rm whisper-asr:latest python -c "import whisper; print('OK')"

# 測試 MiMo Docker
docker run --rm --gpus all mimo-audio:latest python -c "import torch; print('OK')"
```

## 🚀 使用方式

### Whisper 轉錄
```bash
# 方式 1: 使用便捷腳本
./scripts/run_whisper_docker.sh

# 方式 2: 直接使用 Docker
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    whisper-asr:latest \
    python scripts/data_management/transcribe_optical.py \
        --audio-dir examples/optical/spk1 \
        --output data/transcriptions/optical_transcriptions.json \
        --model-size base \
        --language zh
```

### MiMo-Audio ICL 測試
```bash
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    mimo-audio:latest \
    python scripts/data_management/test_icl_with_text.py
```

## 📊 清理前後對比

| 項目 | 清理前 | 清理後 |
|------|--------|--------|
| Host Python 套件 | 50+ 個 ML 套件 | 基本套件 only |
| Host 環境大小 | ~5GB | ~500MB |
| 環境隔離 | ❌ 混亂 | ✅ 完全隔離 |
| 版本衝突風險 | ⚠️ 高 | ✅ 無 |
| 可重複性 | ❌ 依賴 host | ✅ Docker 保證 |
| 遷移難度 | ⚠️ 需重新配置 | ✅ 直接使用映像 |

## ✅ 檢查清單

- [x] 備份當前環境
- [x] 建立 Dockerfile.whisper
- [ ] 構建 whisper-asr:latest 映像
- [ ] 清理 host base 環境
- [ ] 驗證兩個 Docker 環境正常
- [ ] 測試 ICL 功能
- [ ] 更新文檔並提交

## 📅 時間記錄

- **2025-11-13 06:23**: Whisper 轉錄完成 (3456 files)
- **2025-11-13 08:08**: Manifest 生成完成 (100% paired)
- **2025-11-13 [當前]**: 環境清理與 Docker 化進行中

## 🔗 相關檔案

- `Dockerfile.whisper` - Whisper Docker 定義
- `scripts/run_whisper_docker.sh` - Whisper 便捷腳本
- `data/backup_conda_base_env_20251113.txt` - 環境備份
- `data/whisper_packages_to_remove.txt` - 待移除套件清單
