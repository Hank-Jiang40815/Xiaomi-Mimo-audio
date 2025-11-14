# 🚀 實驗快速參考指南

## 最新完成實驗: LoRA Fine-tuning (2025-11-13)

### 基本資訊
- **實驗名稱**: Encoder Fine-tuning with LoRA Rank 32
- **狀態**: ✅ 已完成 (100/100 epochs)
- **時長**: ~61 分鐘
- **輸出目錄**: `./outputs/optical_lora_r32_e100/`

### 核心結果
```
Train Loss: 10.42 → 6.93 (-33.51%)
Val Loss:   7.74 → 6.91 (-10.74%)
Best Val:   6.79 (Epoch 2) 🏆
Status:     已收斂 ✅
```

### 重要檔案
| 檔案 | 用途 |
|------|------|
| `FINETUNE_EXPERIMENT_R32_E100.md` | 📄 完整實驗報告 |
| `outputs/optical_lora_r32_e100/checkpoint_epoch_2.pt` | 🏆 最佳模型 |
| `outputs/optical_lora_r32_e100/checkpoint_epoch_100.pt` | 📦 最終模型 |
| `outputs/optical_lora_r32_e100/training_summary.json` | 📊 評估摘要 |
| `outputs/optical_lora_r32_e100/training.log` | 📝 完整日誌 |

---

## 快速命令

### 檢查訓練狀態
```bash
./check_training_status.sh
```

### 查看訓練日誌
```bash
tail -f ./outputs/optical_lora_r32_e100/training.log
```

### 管理 Tmux Session
```bash
# 列出所有 sessions
tmux ls

# 連接到訓練 session
tmux attach -t finetune_r32_e100

# 關閉 session (訓練已完成，可安全關閉)
tmux kill-session -t finetune_r32_e100
```

### 查看 GPU 狀態
```bash
nvidia-smi
```

---

## 下一步行動

### 🔴 優先級 P0 - 立即執行
- [ ] **修復資料洩漏問題**
  ```bash
  # 重新生成無重疊的 splits
  python create_test_splits.py --no-overlap --verify
  ```

### 🟡 優先級 P1 - 本週內
- [ ] **客觀評估模型**
  ```python
  # 載入最佳 checkpoint
  checkpoint = torch.load("outputs/optical_lora_r32_e100/checkpoint_epoch_2.pt")
  
  # 在 test set 評估
  # - 計算 SI-SDR
  # - 計算 PESQ
  # - 生成增強音訊樣本
  ```

- [ ] **重新訓練 (乾淨資料集)**
  ```bash
  # 使用修復後的 splits + early stopping
  python finetune_encoder.py \
    --lora_rank 32 \
    --epochs 100 \
    --early_stopping_patience 10 \
    --output_dir outputs/optical_lora_r32_clean
  ```

### 🟢 優先級 P2 - 探索性
- [ ] **實驗不同 LoRA rank**
  - Rank 8, 16, 64
  - 比較效能與訓練速度

- [ ] **加入客觀指標作為驗證**
  - 除了 MSE loss，也計算 SI-SDR
  - 作為 early stopping 的依據

---

## 已知問題

### ⚠️ 資料洩漏 (CRITICAL)
```
發現位置: data/splits/finetune_optical/
影響範圍:
  - Train-Val overlap: 196 samples
  - Val-Test overlap: 136 samples
  - Train-Test overlap: 208 samples

影響:
  - 驗證指標不可靠
  - 可能高估模型泛化能力
  - 需重新訓練以獲得真實表現

解決方案:
  1. 重新執行 create_test_splits.py
  2. 加入重疊檢查機制
  3. 驗證新 splits 的完整性
```

### ⚠️ 訓練過長
```
觀察: 最佳 Val loss 在 Epoch 2，但訓練了 100 epochs
原因: 未使用 early stopping
建議: 加入 patience=10 的 early stopping
```

---

## 技術洞察

### ✅ 成功經驗
1. **Flash Attention**: 必須使用 bfloat16
2. **輸入格式**: Mel Spectrogram (128 bands, hop=240)
3. **LoRA 注入**: 使用 `list(named_modules())` 避免迭代修改
4. **Tmux 保護**: 對長時間訓練至關重要

### ⚠️ 注意事項
1. 資料集分割前務必檢查重疊
2. Early stopping 比固定 epochs 更有效
3. 監控 Train-Val gap 避免過擬合
4. 定期保存 checkpoint 以防中斷

---

## 相關文檔索引

### 實驗報告
- [當前實驗完整報告](FINETUNE_EXPERIMENT_R32_E100.md)
- [實驗總日誌](EXPERIMENT_LOG.md)
- [除錯記錄](FINETUNE_ENCODER_DEBUGGING_LOG.md)

### 操作指南
- [Fine-tuning 快速開始](FINETUNE_QUICKSTART.md)
- [完整 Fine-tuning 指南](FINETUNING_GUIDE.md)
- [資料管理指南](data/DATA_MANAGEMENT_GUIDE.md)

### 技術文檔
- [架構分析](FINETUNE_ARCHITECTURE_ANALYSIS.md)
- [音訊增強分析](AUDIO_ENHANCEMENT_ANALYSIS.md)

---

## 實驗配置快速複製

### 當前配置 (LoRA Rank 32)
```python
config = {
    "lora_rank": 32,
    "lora_alpha": 64,
    "batch_size": 8,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "epochs": 100,
    "output_dir": "outputs/optical_lora_r32_e100"
}
```

### 建議改進配置
```python
config = {
    "lora_rank": 32,
    "lora_alpha": 64,
    "batch_size": 8,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "epochs": 100,
    "early_stopping_patience": 10,  # ✅ 新增
    "output_dir": "outputs/optical_lora_r32_clean",
    "clean_splits": True  # ✅ 使用修復後的資料
}
```

---

**最後更新**: 2025-11-13 18:30  
**維護者**: Hank  
**分支**: feat/finetune-encoder
