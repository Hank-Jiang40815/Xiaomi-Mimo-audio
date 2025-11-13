# 資料管理架構使用指南

這個資料管理架構讓您能靈活地管理音訊資料集，用於 ICL 實驗和微調準備。

## 📁 目錄結構

```
data/
├── manifests/              # 資料清單（掃描所有音檔）
│   ├── ldv_manifest.json
│   └── optical_denoised_manifest.json
├── splits/                 # 資料分割配置
│   ├── icl/               # ICL 實驗用
│   │   ├── 10shot.json
│   │   ├── 40shot.json
│   │   ├── 80shot.json
│   │   └── custom_*.json
│   └── finetune/          # 微調用
│       ├── train.json
│       ├── val.json
│       └── test.json
├── sentences/             # 句子內容對應（預留）
└── metadata/              # 元資料（預留）
```

## 📊 資料集說明

### LDV 資料集
- **含噪音音檔 (mix/)**: 828 個
- **乾淨音檔 (spk/)**: 416 個
- **命名格式**: `boy1_papercup_LDV_001.wav`
- **說明**: 每個乾淨音檔對應約 2 個不同噪音版本

### Optical_denoised 資料集
- **含噪音音檔 (mix/)**: 828 個  
- **乾淨音檔 (spk1/)**: 416 個
- **命名格式**: `boy1_WOLDVlean_001.wav`
- **說明**: 經過降噪處理的版本

### 重要概念
- **句子編號 (001, 002, ...)**: 相同編號代表相同句子內容
- **語者 (boy1, boy2, ...)**: 代表不同的說話者
- **配對關係**: 每個 noisy 檔案會嘗試配對對應的 clean 檔案（透過句子編號）

---

## 🚀 快速開始

### 步驟 1: 建立 Manifest（資料清單）

Manifest 檔案包含所有音檔的完整資訊和配對關係。

#### LDV 資料集
```bash
python scripts/data_management/create_manifest.py \
    --dataset ldv \
    --noisy-dir examples/ldv/mix \
    --clean-dir examples/ldv/spk \
    --output data/manifests/ldv_manifest.json
```

#### Optical_denoised 資料集
```bash
python scripts/data_management/create_manifest.py \
    --dataset optical_denoised \
    --noisy-dir examples/optical_denoised/mix \
    --clean-dir examples/optical_denoised/spk1 \
    --output data/manifests/optical_denoised_manifest.json
```

**輸出範例：**
```json
{
  "dataset_name": "ldv",
  "total_samples": 828,
  "matched_pairs": 416,
  "unmatched_samples": 412,
  "samples": [
    {
      "id": "001",
      "sentence_id": "001",
      "speaker": "boy1",
      "noise_type": "papercup",
      "noisy_file": "examples/ldv/mix/boy1_papercup_LDV_001.wav",
      "clean_file": "examples/ldv/spk/boy1_papercup_LDV_001.wav",
      "text": "",
      "duration": null,
      "noise_level": null
    },
    ...
  ]
}
```

### 步驟 2: 選擇資料分割 (Split)

有了 manifest 後，就可以靈活選擇要用哪些樣本。

---

## 📖 ICL 實驗用法

### 2.1 順序選擇 N 個樣本

選擇前 40 個樣本（001-040）：
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl \
    --shots 40 \
    --method sequential \
    --output data/splits/icl/40shot.json
```

常用的 shot 數量：
```bash
# 10-shot
--shots 10 --output data/splits/icl/10shot.json

# 40-shot (Exp A+ 使用的配置)
--shots 40 --output data/splits/icl/40shot.json

# 80-shot
--shots 80 --output data/splits/icl/80shot.json

# 160-shot
--shots 160 --output data/splits/icl/160shot.json
```

### 2.2 自訂範圍選擇

**範例 1**: 選擇 1-20, 50-60, 100-109 (共 31 個)
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl \
    --custom "1-20,50-60,100-109" \
    --output data/splits/icl/custom_31shot.json
```

**範例 2**: 只選擇特定編號 1, 5, 10, 15, 20
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl \
    --custom "1,5,10,15,20" \
    --output data/splits/icl/custom_5shot.json
```

**範例 3**: 選擇 1-10 和 100-200
```bash
--custom "1-10,100-200"
```

### 2.3 隨機選擇

隨機選擇 40 個樣本（可重現）：
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl \
    --shots 40 \
    --method random \
    --seed 42 \
    --output data/splits/icl/40shot_random.json
```

---

## 🔧 微調資料分割用法

### 3.1 基本分割（8:1:1）

將資料分成 訓練:驗證:測試 = 80%:10%:10%
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode finetune \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --output-dir data/splits/finetune/
```

**輸出**：
- `data/splits/finetune/train.json` (約 662 個樣本)
- `data/splits/finetune/val.json` (約 83 個樣本)
- `data/splits/finetune/test.json` (約 83 個樣本)

### 3.2 自訂比例

90% 訓練，5% 驗證，5% 測試：
```bash
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode finetune \
    --train-ratio 0.9 \
    --val-ratio 0.05 \
    --test-ratio 0.05 \
    --output-dir data/splits/finetune_90_5_5/
```

---

## 💡 實際使用情境

### 情境 1: 測試不同 shot 數量的 ICL 效果

建立多個不同 shot 數量的配置：
```bash
# 建立 10/40/80/160-shot 配置
for shots in 10 40 80 160; do
    python scripts/data_management/split_selector.py \
        --manifest data/manifests/ldv_manifest.json \
        --mode icl --shots $shots --method sequential \
        --output data/splits/icl/${shots}shot.json
done
```

### 情境 2: 準備微調實驗

```bash
# 1. 建立 manifest
python scripts/data_management/create_manifest.py \
    --dataset ldv \
    --noisy-dir examples/ldv/mix \
    --clean-dir examples/ldv/spk \
    --output data/manifests/ldv_manifest.json

# 2. 分割資料集
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode finetune \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --seed 42 \
    --output-dir data/splits/finetune/

# 3. 檢視分割結果
cat data/splits/finetune/train.json | jq '.total_samples'
cat data/splits/finetune/val.json | jq '.total_samples'
cat data/splits/finetune/test.json | jq '.total_samples'
```

### 情境 3: 只想用特定句子做實驗

假設您知道某些句子特別適合測試：
```bash
# 選擇句子編號 1-10, 50-60, 200-210
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl \
    --custom "1-10,50-60,200-210" \
    --output data/splits/icl/custom_experiment1.json
```

---

## 📝 Split 檔案格式

### ICL Split 範例
```json
{
  "dataset": "ldv",
  "mode": "icl",
  "total_samples": 40,
  "method": "sequential",
  "range": "1-40",
  "seed": null,
  "samples": [
    {
      "id": "001",
      "sentence_id": "001",
      "speaker": "boy1",
      "noisy_file": "examples/ldv/mix/boy1_papercup_LDV_001.wav",
      "clean_file": "examples/ldv/spk/boy1_papercup_LDV_001.wav",
      ...
    },
    ...
  ]
}
```

### 微調 Split 範例
```json
{
  "dataset": "ldv",
  "mode": "finetune",
  "split": "train",
  "total_samples": 662,
  "samples": [ ... ]
}
```

---

## 🔍 進階用法

### 檢視 Manifest 資訊

```bash
# 檢視總樣本數
cat data/manifests/ldv_manifest.json | jq '.total_samples'

# 檢視配對情況
cat data/manifests/ldv_manifest.json | jq '{matched: .matched_pairs, unmatched: .unmatched_samples}'

# 檢視前 5 個樣本
cat data/manifests/ldv_manifest.json | jq '.samples[:5]'

# 搜尋特定句子編號
cat data/manifests/ldv_manifest.json | jq '.samples[] | select(.sentence_id == "041")'
```

### 檢視 Split 資訊

```bash
# 檢視 split 的樣本數量
cat data/splits/icl/40shot.json | jq '.total_samples'

# 列出所有句子編號
cat data/splits/icl/40shot.json | jq '[.samples[].sentence_id]'

# 檢查是否有對應的 clean 檔案
cat data/splits/icl/40shot.json | jq '[.samples[] | select(.clean_file != null)] | length'
```

---

## ❓ 常見問題

### Q1: 為什麼 LDV 有 828 個 noisy 但只有 416 個 clean？
**A**: 每個乾淨音檔可能對應多個不同噪音版本。Manifest 會自動配對，配對成功的會有 `clean_file`，未配對的會是 `null`。

### Q2: 如何知道哪些樣本有配對成功？
**A**: 
```bash
# 查看配對統計
cat data/manifests/ldv_manifest.json | jq '{matched: .matched_pairs, unmatched: .unmatched_samples}'

# 列出所有有配對的樣本
cat data/manifests/ldv_manifest.json | jq '.samples[] | select(.clean_file != null)'
```

### Q3: 自訂範圍的語法是什麼？
**A**: 
- 單一數字: `"1,5,10"`
- 範圍: `"1-10"` (包含 1 和 10)
- 混合: `"1-10,20,30-40"`
- 空格會被忽略: `"1-10, 20, 30-40"` 也可以

### Q4: 如何確保實驗可重現？
**A**: 使用 `--seed` 參數：
```bash
--seed 42
```

### Q5: 可以用在 quick_inference_expA++.sh 嗎？
**A**: 可以！後續會整合 split 檔案讀取功能。目前的 split 檔案已經包含所有需要的資訊（noisy_file, clean_file 路徑）。

---

## 🎯 下一步

1. **句子內容管理** (預留)
   - 建立 `data/sentences/sentence_mapping.json`
   - 編號 → 句子內容的對應表

2. **噪音等級分組** (預留)
   - 支援按 x60/x65/x70 分組
   - 測試不同噪音等級的效果

3. **整合到推論工具**
   - 更新 `quick_inference_expA++.sh` 
   - 支援讀取 split 檔案

4. **整合到微調工具**
   - 更新 `finetune_encoder.py`
   - 直接讀取 split 檔案進行訓練

---

## 📚 工具參考

### create_manifest.py
建立完整的資料清單，掃描音檔並建立配對關係。

**必要參數**:
- `--dataset`: 資料集名稱
- `--noisy-dir`: 含噪音音檔目錄
- `--output`: 輸出 manifest 路徑

**可選參數**:
- `--clean-dir`: 乾淨音檔目錄
- `--compute-duration`: 計算音檔長度

### split_selector.py
靈活選擇資料分割，用於 ICL 或微調。

**ICL 模式**:
- `--mode icl`
- `--shots N`: 選擇 N 個樣本
- `--method`: sequential 或 random
- `--custom`: 自訂範圍字串
- `--output`: 輸出檔案

**微調模式**:
- `--mode finetune`
- `--train-ratio`: 訓練集比例
- `--val-ratio`: 驗證集比例  
- `--test-ratio`: 測試集比例
- `--output-dir`: 輸出目錄

---

**需要協助？** 請查看工具的 `--help`：
```bash
python scripts/data_management/create_manifest.py --help
python scripts/data_management/split_selector.py --help
```
