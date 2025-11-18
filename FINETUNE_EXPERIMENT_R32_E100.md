# 🧪 Encoder Fine-tuning 實驗報告 - LoRA Rank 32

## 2025-11-18｜V2-Norm：Codebook Alignment + Waveform Normalization

| 項目 | 設定 |
|------|------|
| **訓練腳本** | `bash start_training_v2.sh`（啟用 `--normalize-waveform`，tmux `finetune_v2`） |
| **Loss** | Feature MSE + Codebook Alignment L1 + VQ Commit (`lambda_feat=1, lambda_code=1, lambda_vq=0.1`) |
| **輸出** | `outputs/optical_lora_v2_norm_r32_e100/` (`best_model.pt`, checkpoint every 10 epochs, `training_history.json`) |
| **推論** | `CHECKPOINT=outputs/optical_lora_v2_norm_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2_norm/enhanced_050.wav ./test_inference_docker.sh --no-tmux` |

### 📊 訓練指標
| 指標 | Epoch 1 | Epoch 100 | Best |
|------|---------|-----------|------|
| **Train Loss** | 6.06 | 0.06 | 0.06 (E100) |
| **Val Loss** | 3.14 | 0.04 | **0.033 (E47)** |

- 在轉 Mel 之前做 waveform z-score，loss 尺度顯著下降，最佳驗證落在第 47 epoch，後期維持 0.04 左右。  
- `training_history.json` 可看到 train/val 曲線幾乎重疊，顯示 normalization 提升穩定度。  
- 新增的 inference 輸出 `outputs/test_inference_v2_norm/enhanced_050.wav`（副本 `examples/codebook_align_inference/boy1_WOLDV_050_enhanced_v2_norm.wav`）可與舊版本對照。

### 📝 補充紀錄
1. **Dataset 正規化旗標**：在 `OpticalDataset` 中加入 waveform z-score（mean=0、std=1），並透過 `--normalize-waveform` 控制；`start_training_v2.sh` 預設開啟。  
2. **tmux/Docker 流程**：與 V2 相同，輸出 log 為 `outputs/optical_lora_v2_norm_r32_e100/training.log`。  
3. **後續工作**：需要在乾淨 splits 上重新驗證，並計算客觀指標確認音質是否真的提升。  
4. **樣本管理**：所有 inference wav 仍透過 Git LFS 追蹤，集中於 `examples/codebook_align_inference/`。

---

## 2025-11-17｜V2：Codebook Alignment + Inference Sanity Check

| 項目 | 設定 |
|------|------|
| **訓練腳本** | `bash start_training_v2.sh`（docker + tmux `finetune_v2`） |
| **Loss** | Feature MSE + Codebook Alignment L1 + VQ Commit (`lambda_feat=1, lambda_code=1, lambda_vq=0.1`) |
| **輸出** | `outputs/optical_lora_v2_r32_e100/` (`best_model.pt`, checkpoint 每 10 epochs, `training_history.json`) |
| **推論** | `CHECKPOINT=outputs/optical_lora_v2_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2/enhanced_050.wav ./test_inference_docker.sh --no-tmux` |

### 📊 訓練指標
| 指標 | Epoch 1 | Epoch 100 | Best |
|------|---------|-----------|------|
| **Train Loss** | 12.08 | 8.22 | 8.22 (E100) |
| **Val Loss** | 9.23 | 8.20 | **8.06 (E2)** |

- 100 epochs 全程穩定，最佳驗證仍出現在前期（E2），顯示可加入 early stopping。  
- Codebook Alignment Loss 未發散，train/val 曲線平行（詳見 `training_history.json`）。  
- `best_model.pt` 已用於單檔 inference，輸出 `outputs/test_inference_v2/enhanced_050.wav`（長度 2.6 秒）。

### 📝 補充紀錄
1. **Quantizer dtype 修正**：`finetune_encoder_v2.py` 內改為從 quantizer buffer 取得 dtype，避免沒有梯度參數時 crash。  
2. **執行流程**：`start_training_v2.sh` 會檢查資料/模型、建立 tmux、在 docker 中啟動 `python finetune_encoder_v2.py`。log 透過 `tee outputs/optical_lora_v2_r32_e100/training.log` 留存。  
3. **Inference output**：`enhanced_050.wav` 可與 `examples/optical/mix/boy1_WOLDV_050.wav` 對照；後續需批次化評估 (SI-SDR / PESQ)。  
4. **TODO**：重建無洩漏 splits、加入 early stopping、撰寫客觀測試腳本、整理主觀聽感。

---

## 2025-11-13｜V1：LoRA Rank 32, 100 Epochs（舊實驗）

**實驗日期**: 2025-11-13  
**實驗目的**: 使用 LoRA 微調 MiMo-Audio-Tokenizer 編碼器，提升 Optical 麥克風音訊降噪能力  
**實驗狀態**: ✅ 已完成  

---

## 📋 實驗配置

### 模型架構
- **Base Model**: MiMo-Audio-Tokenizer (1.3B parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank: 32
  - Alpha: 64
  - Target Modules: 192 layers (attn.qkv, attn.proj, mlp.fc1, mlp.fc2)
  - Trainable Parameters: **678,772,802 / 1,308,246,082 (51.88%)**

### 訓練設定
```python
{
  "batch_size": 8,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 64,
  "learning_rate": 5e-5,
  "optimizer": "AdamW",
  "epochs": 100,
  "dtype": "bfloat16",  # Flash Attention 需求
  "device": "cuda (RTX 5090, 32GB)"
}
```

### 資料集
```
總樣本數: 3,456
- Train: 2,764 (80.0%)
- Val:   345 (10.0%)
- Test:  347 (10.0%)

說話者: 12 位 (6 男, 6 女)
- 每位說話者: 288 樣本 (完美平衡)
```

**⚠️ 重要發現**: 資料集存在洩漏問題
- Train-Val overlap: 196 samples
- Val-Test overlap: 136 samples
- Train-Test overlap: 208 samples

此問題影響驗證指標的可靠性，需在後續實驗中修復。

### 輸入格式
- **特徵**: Mel Spectrogram (非原始波形)
- **參數**:
  - n_mels: 128
  - n_fft: 1024
  - hop_length: 240
  - sample_rate: 24000 Hz

---

## 📊 訓練結果

### Loss 改善
| 指標 | Epoch 1 | Epoch 100 | Best | 改善幅度 |
|------|---------|-----------|------|----------|
| **Train Loss** | 10.4220 | 6.9297 | 6.9297 (E100) | **-33.51%** |
| **Val Loss** | 7.7418 | 6.9106 | 6.7924 (E2) | **-10.74%** |

### 收斂行為分析

#### 分階段改善
| 階段 | Train 改善 | Val 改善 | 觀察 |
|------|-----------|----------|------|
| **前 20 epochs** | -32.48% | -9.72% | 🚀 快速學習期 |
| **中段 (21-60)** | -0.75% | -0.82% | 📊 平穩改善期 |
| **後段 (61-100)** | -0.41% | -0.36% | 📉 接近飽和 |
| **最後 10 epochs** | -0.18% | -0.07% | ✅ 已收斂 |

#### 關鍵發現
1. **快速收斂**: 主要改善在前 20 epochs 完成
2. **穩定訓練**: Train-Val gap 僅 0.28%，無過擬合
3. **已達飽和**: 最後 20 epochs 標準差 < 0.003
4. **最佳點早現**: 最佳 Val loss 出現在 Epoch 2

---

## 🎯 實驗評估

### ✅ 成功點
1. **訓練穩定**: 100 epochs 順利完成，無中斷
2. **無過擬合**: Train-Val gap 極小 (0.28%)
3. **收斂良好**: Loss 曲線平滑，無震盪
4. **資源高效**: GPU 利用率 89-90%，速度 ~6 it/s

### ⚠️ 問題點
1. **資料洩漏**: 發現 train/val/test splits 有重疊，影響驗證可靠性
2. **過度訓練**: 100 epochs 可能過多，前 20-30 epochs 已足夠
3. **早期停止失效**: 最佳 Val loss 在 Epoch 2，但持續訓練到 100

### 📈 改善空間
| 項目 | 當前狀態 | 改善方向 |
|------|---------|---------|
| **資料品質** | 有洩漏問題 | ✅ 重新生成無重疊 splits |
| **訓練長度** | 100 epochs | ✅ 使用 early stopping (patience=10) |
| **Checkpoint 策略** | 每個 epoch 儲存 | ✅ 只保存最佳 + 最後 3 個 |
| **驗證策略** | 僅用 Val loss | ✅ 加入客觀指標 (SI-SDR, PESQ) |

---

## 🔬 技術洞察

### 1. 輸入格式的重要性
**問題**: 最初使用原始波形 (waveform) 導致 shape mismatch  
**解決**: 改用 Mel Spectrogram，符合 tokenizer 預期輸入  
**教訓**: 仔細閱讀 `config.json` 中的 feature extractor 設定

### 2. Flash Attention 的限制
**問題**: 使用 fp32 會觸發 RuntimeError  
**解決**: 強制使用 bfloat16 dtype  
**教訓**: 特定優化技術有嚴格的型別要求

### 3. LoRA 注入時機
**問題**: 在 `named_modules()` 迭代中修改 module 導致 RuntimeError  
**解決**: 先用 `list()` 收集所有 modules，再逐一注入  
**教訓**: Python 迭代器不允許在迭代中修改結構

### 4. 訓練長度選擇
**觀察**: 前 20 epochs 完成 32% 改善，後 80 epochs 僅 1% 改善  
**建議**: 使用 early stopping，避免浪費計算資源  
**教訓**: 更多 epochs ≠ 更好結果，需監控邊際效益

---

## 📦 輸出檔案

### Checkpoints
```
outputs/optical_lora_r32_e100/
├── checkpoint_epoch_2.pt        # 🏆 最佳 Val loss (6.7924)
├── checkpoint_epoch_100.pt      # 最終模型
├── checkpoint_epoch_*.pt        # 其他 98 個 checkpoints
├── training.log                 # 完整訓練日誌
└── training_summary.json        # 評估結果摘要
```

### 建議使用
- **推論/部署**: `checkpoint_epoch_2.pt` (最佳泛化能力)
- **繼續訓練**: `checkpoint_epoch_100.pt` (最新狀態)

---

## 🚀 後續行動計畫

### 優先級 P0 (立即執行)
- [ ] **修復資料洩漏問題**
  - 重新生成 train/val/test splits (無重疊)
  - 驗證 split 完整性
  - 更新 `data/splits/finetune_optical/` 檔案

### 優先級 P1 (本週內)
- [ ] **在乾淨資料集上重新訓練**
  - 使用修復後的 splits
  - 加入 early stopping (patience=10)
  - 比較修復前後的差異

- [ ] **客觀評估模型表現**
  - 在 test set 上計算 SI-SDR
  - 計算 PESQ 分數
  - 進行主觀聽感測試

### 優先級 P2 (探索性)
- [ ] **測試不同 LoRA 配置**
  - Rank 8 vs 16 vs 32 vs 64
  - 只 fine-tune attention vs 全部層

- [ ] **實驗不同訓練策略**
  - Learning rate schedule (cosine decay)
  - Warmup steps
  - 不同 batch size

---

## 📝 實驗日誌

### 2025-11-13 17:15 - 訓練開始
- 配置: LoRA rank 32, batch 8, 100 epochs
- 使用 tmux session 保護 (finetune_r32_e100)
- GPU: RTX 5090 (32GB)

### 2025-11-13 18:16 - 訓練完成
- 總時長: ~61 分鐘
- 最終 Train Loss: 6.9297
- 最終 Val Loss: 6.9106
- 最佳 Val Loss: 6.7924 (Epoch 2)

### 2025-11-13 18:30 - 評估完成
- 發現資料洩漏問題
- 生成訓練摘要
- 建立實驗報告

---

## 💡 經驗教訓總結

### 技術層面
1. ✅ **Flash Attention 需要 bfloat16**
2. ✅ **Mel Spectrogram 是正確的輸入格式**
3. ✅ **LoRA 注入要避免在迭代中修改結構**
4. ✅ **Tmux 保護對長時間訓練很重要**

### 實驗設計層面
1. ⚠️ **資料集分割需嚴格檢查重疊**
2. ⚠️ **Early stopping 比固定 epochs 更好**
3. ⚠️ **驗證集 loss 可能在早期就達最佳**
4. ⚠️ **更多 epochs 不一定更好**

### 流程改善
1. 📋 **訓練前檢查清單**: 資料完整性、配置驗證、資源確認
2. 📊 **監控指標**: 除了 loss，也要加入客觀音訊指標
3. 💾 **Checkpoint 管理**: 只保存必要的，避免磁碟空間浪費
4. 📝 **實驗記錄**: 即時記錄，包含失敗嘗試

---

## 🔗 相關文檔

- [Fine-tuning 快速開始指南](FINETUNE_QUICKSTART.md)
- [除錯完整記錄](FINETUNE_ENCODER_DEBUGGING_LOG.md)
- [訓練腳本](finetune_encoder.py)
- [資料集分析](data/DATA_MANAGEMENT_GUIDE.md)

---

**實驗負責人**: Hank  
**最後更新**: 2025-11-13 18:30  
**狀態**: ✅ 完成 (需修復資料洩漏後重訓)
