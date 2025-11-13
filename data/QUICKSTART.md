# 🚀 快速開始 - 資料管理架構

## 30 秒快速測試

```bash
# 1. 建立 LDV 資料集清單
python scripts/data_management/create_manifest.py \
    --dataset ldv \
    --noisy-dir examples/ldv/mix \
    --clean-dir examples/ldv/spk \
    --output data/manifests/ldv_manifest.json

# 2. 建立 40-shot ICL 配置
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl --shots 40 --method sequential \
    --output data/splits/icl/40shot.json

# 3. 查看結果
cat data/splits/icl/40shot.json | jq '.total_samples'
```

## 📖 完整說明

請參考: `data/DATA_MANAGEMENT_GUIDE.md`

## 🎯 常用命令

### ICL 實驗

```bash
# 10/40/80/160-shot
for shots in 10 40 80 160; do
    python scripts/data_management/split_selector.py \
        --manifest data/manifests/ldv_manifest.json \
        --mode icl --shots $shots \
        --output data/splits/icl/${shots}shot.json
done
```

### 自訂範圍

```bash
# 選擇 1-20, 50-60, 100
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode icl --custom "1-20,50-60,100" \
    --output data/splits/icl/my_experiment.json
```

### 微調準備

```bash
# 8:1:1 分割
python scripts/data_management/split_selector.py \
    --manifest data/manifests/ldv_manifest.json \
    --mode finetune \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --output-dir data/splits/finetune/
```

## 📊 資料集統計

- **LDV**: 828 個含噪音音檔, 416 個乾淨音檔 (576 個已配對)
- **句子編號**: 001-828 (相同編號 = 相同句子內容)
- **語者**: boy1, boy2, ...

## ❓ 需要協助？

```bash
python scripts/data_management/create_manifest.py --help
python scripts/data_management/split_selector.py --help
```
