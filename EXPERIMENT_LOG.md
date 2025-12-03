# MiMo-Audio 實驗記錄

## 最新實驗 (2025-11-26)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=4, E100)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100 \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q4 \
       --epochs 100 --batch-size 4 --lr 1e-4 --num-quantizer-layers 4\""
  ```

**架構與手法**:
- 凍結：MiMo-Audio-Tokenizer 的 encoder + RVQ quantizer + decoder。
- CodeRefiner：單一共享 Transformer (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1)，作用於離散 codes。
- 資料流程：
  - noisy/clean waveform → Mel（依 tokenizer config）→ encoder.get_features → quantizer → 得到 `codes_noisy`, `codes_clean`（形狀約 [n_q, T]）。
  - 這次使用前 4 個 RVQ 層（N=4），對每一層 q：
    - `logits_q = Refiner(codes_noisy[q])`，CE 對齊 `codes_clean[q]`。
  - 總 loss 為 4 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q4/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.6557 @ epoch 90**；Final Val Loss: 2.6560
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q4/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q4/inference_boy1_001_refined.wav
  ```

**觀察 / 待辦**:
1. 多層 (N=4) CE loss 穩定收斂至 ~2.66，優於單層版本（ppl 約從 ~16 降至 ~14 左右），但仍有不小 gap，說明 noisy→clean code 映射仍具不確定性。
2. 需搭配主觀聽感與客觀指標（SI-SDR/PESQ/STOI）比較：
   - Base tokenizer（無 refinement）
   - 單層 CodeRefiner（outputs/code_refiner_optical_e100）
   - 多層 CodeRefiner（本實驗）
3. 若聽感顯示有改善，可考慮：
   - 擴展到更多 RVQ 層或加入輕量聲學 loss（小權重 MR-STFT / SDR）。
   - 加入 speaker/noise conditioning，讓映射更可控，而不是僅靠 codes pattern。

---

## 最新實驗 (2025-12-03)
### 🧪 CodeRefiner-A - 多層 RVQ Token Refinement (N=8, E100, WaveNorm + Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement-A-layer-sweep`
- 訓練腳本：`finetune_code_refiner.py`（單一 head 版本）
- tmux + docker：
  ```bash
  tmux new -s code_refiner_A_e100_q8_norm_officialmel \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_A_e100_q8_norm_officialmel \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 8 \
       --normalize-waveform \
       --official-mel\""
  ```

**架構與手法**:
- 與官方前處理 N=8 實驗相同：凍結 tokenizer，使用共享 Transformer CodeRefiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 8 個 RVQ 層做 code-level CE 對齊。
- 前處理：
  - WaveNorm：訓練時在 Dataset 中對 noisy/clean waveform 做 per-utterance `(x-mean)/std`。
  - Official Mel：waveform → config.nfft/window_size 的 Mel → log-mel（`--official-mel`）。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_A_e100_q8_norm_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.2423 @ epoch 69**；Final Val Loss: 2.2447（略高於 N=4_norm_officialmel ≈1.98，約略接近 per-layer head 版 N=8_heads≈2.24）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_A_e100_q8_norm_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_A_e100_q8_norm_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 8 \
    --normalize-waveform \
    --official-mel
  ```

**觀察 / 待辦**:
1. 在 A 線（單 head）下，N=8_norm_officialmel 的 Val CE 約 2.24，明顯低於舊 N=8（非 official-mel、非 WaveNorm）的 ~2.78，但仍高於 N=4_norm_officialmel 的 ~1.98，顯示「只修前 8 層」的難度仍高於只修前 4 層。
2. 與 B 線 per-layer head 的 N=8_heads（Best≈2.2430）相比，A 線的單 head 結果非常接近，說明在目前設定下 head 拆層與否並非主導因素。
3. 實務上仍可將 N=4_norm_officialmel 視為主工作點，N=8_norm_officialmel 作為「多修幾層是否有實質聽感提升」的對照實驗，需搭配主觀聽感進一步判斷是否值得增加這層複雜度。

---
## 最新實驗 (2025-12-01)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=4, E100, Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q4_officialmel \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q4_officialmel \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 4 \
       --official-mel\""
  ```

**架構與手法**:
- 與 baseline N=4 相同：凍結 MiMo-Audio-Tokenizer 的 encoder + RVQ quantizer + decoder，使用共享 Transformer CodeRefiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 4 個 RVQ 層的離散 codes 做 CE 對齊 (noisy→clean)。
- 差異點：改用 **MiMo 官方 wav2mel 前處理** 產生 codes：
  - waveform → `MelSpectrogram`(n_fft=config.nfft, win_length=config.window_size, hop_length=config.hop_length, f_min=config.fmin, f_max=config.fmax, n_mels=config.n_mels, power=1.0, center=True) → log-mel。
  - training 與 inference 都透過 `--official-mel` 使用同一條路徑。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q4_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.0489 @ epoch 27**；Final Val Loss: 2.0610（相較原始 N=4 約 2.6557，有明顯改善）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q4_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q4_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 4 \
    --official-mel
  ```

**觀察 / 待辦**:
1. 使用官方 wav2mel (nfft=960 + log-mel) 後，N=4 CodeRefiner 的 Val CE 從 ~2.66 進一步降到 ~2.05 左右，顯示與 MiMo 預訓練時一致的前處理，確實有助於 noisy→clean code 對齊任務的可學性。
2. 推理端為避免 RVQ decode 出現 out-of-range index 問題，增加了 per-layer clamping：在將 refined codes 丟回 `tokenizer.encoder.decode_vq` 前，會依照每一層的 codebook size 將 indices 限制在合法範圍（不影響訓練 loss，但提升推理穩定度）。
3. 建議後續將「N=4 + official-mel」視為 CodeRefiner 主線的預設前處理，WaveNorm 版則保留為負面對照；下一步可在此設定下再嘗試小權重聲學 loss 或 conditioning 設計。

---

## 最新實驗 (2025-12-02)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=4, E100, WaveNorm + Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q4_norm_officialmel \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q4_norm_officialmel \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 4 \
       --normalize-waveform \
       --official-mel\""
  ```

**架構與手法**:
- 與 N=4_officialmel 相同：凍結 tokenizer，使用共享 Transformer Refiner 對前 4 個 RVQ 層做 code-level CE 對齊。
- 差異點：在 waveform 端啟用 per-utterance normalization，再接官方 wav2mel (config.nfft + log-mel)。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q4_norm_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **1.9814 @ epoch 36**；Final Val Loss: 1.9900（比 N=4_officialmel ≈2.05 再略好一些）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q4_norm_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q4_norm_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 4 \
    --official-mel
  ```

**觀察 / 待辦**:
1. 在「已經使用官方 wav2mel」的前提下再加上 waveform normalize，Val CE 從 ~2.05 進一步降到 ~1.98，幅度不大但方向是正面，顯示在此設定下 WaveNorm 不再是明顯負面因素。
2. 目前 N=4_officialmel_norm 是所有 CodeRefiner 配置中 CE 最低的一個，建議以此作為後續加上 conditioning / 聲學 loss 的起點，並以 N=4_officialmel 作為對照。
3. 尚未在推理流程中對 waveform 做同樣 normalize（當前 inference 只使用官方 wav2mel），若要完全對齊訓練前處理，可在未來加上對 waveform 的可選 norm flag 做進一步 ablation。

---

## 最新實驗 (2025-11-28)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=4, E100, WaveNorm)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q4_norm \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q4_norm \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 4 \
       --normalize-waveform\""
  ```

**架構與手法**:
- 與原本 N=4 實驗相同：凍結 MiMo-Audio-Tokenizer 的 encoder + RVQ quantizer + decoder，使用共享 Transformer CodeRefiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 4 個 RVQ 層的離散 codes 做 CE 對齊 (noisy→clean)。
- 差異點：在 `OpticalCodeDataset` 中新增 per-utterance waveform normalization：
  - noisy/clean waveform 在轉 Mel 前先做 `(x - mean) / std`（std ≥ 1e-6），其餘 Mel 參數維持與舊實驗相同。
- 資料流程：
  - normalized noisy/clean waveform → Mel → encoder.get_features → quantizer → `codes_noisy`, `codes_clean`（約 [20, T]）。
  - 使用前 4 個 RVQ 層（N=4），逐層計算 CE(noisy_q→clean_q)，總 loss 取 4 層平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q4_norm/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.7965 @ epoch 78**；Final Val Loss: 2.7987（略高於未做 WaveNorm 的 N=4 實驗）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q4_norm/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q4_norm/inference_boy1_001_refined.wav \
    --num-quantizer-layers 4
  ```

**觀察 / 待辦**:
1. 加入 per-utterance waveform normalization 後，N=4 的 Val CE 從 ~2.66 稍微上升到 ~2.80，說明在「code-level CE」任務下，這種 normalization 並沒有帶來明顯的收斂改善，甚至略微不利。
2. 需要搭配聽感與客觀指標（SI-SDR/PESQ/STOI）比較 `code_refiner_optical_e100_q4` vs `code_refiner_optical_e100_q4_norm`，確認 waveform normalization 對實際音質是否有正面或負面影響；目前從 CE 角度看，baseline N=4 仍略優。
3. 若聽感也證實 WaveNorm 版沒有優勢，後續可以將 CodeRefiner 線的主設定維持在「不做 amplitude normalize」，把精力放在 conditioning（speaker/noise）與聲學 loss 的設計，而不是再調整前處理。

---

## 最新實驗 (2025-11-26)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=8, E100)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q8 \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q8 \
       --epochs 100 --batch-size 4 --lr 1e-4 --num-quantizer-layers 8\""
  ```

**架構與手法**:
- 與 N=4 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2) 對前 N=8 個 RVQ 層做 code-level CE 對齊。
- 對每層 q∈{0…7} 個別算 CE(noisy_q→clean_q)，總 loss 為 8 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q8/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.7768 @ epoch 92**；Final Val Loss: 2.7773
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q8/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q8/inference_boy1_001_refined.wav \
    --num-quantizer-layers 8
  ```

**觀察 / 待辦**:
1. 將 RVQ 層數從 4 擴展到 8，Val CE 稍有改善（2.65→2.78 區間內），但變化不劇烈，顯示額外層的可修正空間有限。
2. 需要與 Base / N=1 / N=4 版本進行主觀與 SI-SDR/PESQ/STOI 對照，確認多層 refinement 是否帶來可感知的音質提升。
3. 若提升有限，未來可能改為「選擇性 refinement」（只修最敏感的幾層），或將重心轉向加入 speaker/noise conditioning 與聲學 loss。

---

## 最新實驗 (2025-12-01)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=8, E100, Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 排程（節錄單次指令）：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q8_officialmel \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 8 \
      --official-mel"
  ```

**架構與手法**:
- 與 baseline N=8 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 8 個 RVQ 層做 code-level CE 對齊。
- 差異點：改用 MiMo 官方 wav2mel（config.nfft + log-mel）產生 codes，訓練與推理皆透過 `--official-mel`。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q8_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.0484 @ epoch 35**；Final Val Loss: 2.0530（相較原 N=8 約 2.7768，有明顯改善）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q8_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q8_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 8 \
    --official-mel
  ```

**觀察 / 待辦**:
1. 在 N=8 的設定下，official-mel 將 Val CE 從 ~2.78 壓到 ~2.05，幅度與 N=4 類似，顯示 align 到 MiMo 原生前處理對多層 RVQ code refinement 也有實質幫助。
2. 由於 N=8 本身已經比 N=4 難學，下一步可先在 N=4 版本上優先嘗試聲學 loss / conditioning，再視需要擴展到 N=8 官方前處理版本。
3. 後續分析建議以 N=4_officialmel / N=8_officialmel 作為主線，WaveNorm 與舊前處理版本保留為對照。

---

## 最新實驗 (2025-11-28)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=8, E100, WaveNorm)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q8_norm \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q8_norm \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 8 \
       --normalize-waveform\""
  ```

**架構與手法**:
- 與原本 N=8 實驗相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 8 個 RVQ 層做 code-level CE 對齊。
- 差異點：在 `OpticalCodeDataset` 啟用 per-utterance waveform normalization，noisy/clean waveform 在轉 Mel 前先做 `(x - mean) / std`，其餘 Mel 與 tokenizer/quantizer 設定維持不變。
- 資料流程：
  - normalized noisy/clean waveform → Mel → encoder.get_features → quantizer → `codes_noisy`, `codes_clean`（約 [20, T]）。
  - 使用前 8 個 RVQ 層（N=8），逐層 CE(noisy_q→clean_q)，總 loss 為 8 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q8_norm/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.0477 @ epoch 89**；Final Val Loss: 3.0480（明顯高於未做 WaveNorm 的 N=8 實驗，後者約為 2.78）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q8_norm/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q8_norm/inference_boy1_001_refined.wav \
    --num-quantizer-layers 8
  ```

**觀察 / 待辦**:
1. 對 N=8 而言，加入 per-utterance waveform normalization 後，Val CE 由 ~2.78 上升到 ~3.05，退化幅度比 N=4 更明顯，顯示在多層 RVQ token refinement 任務裡，WaveNorm 並沒有幫助 code-level CE 收斂，甚至削弱了可學訊息。
2. 待進一步主觀聽感與客觀指標比較 `code_refiner_optical_e100_q8` vs `code_refiner_optical_e100_q8_norm`，確認音質是否同樣退化；若無明顯優勢，可考慮在 CodeRefiner 主線中維持「不做 waveform normalize」的設定。
3. 綜合 N=4/N=8 的結果，WaveNorm 對 CodeRefiner 的 CE 表現皆無正向效果，後續若要導入 normalization，可能需要改在 Mel / feature 層面重新設計，而不是直接對 waveform 做 per-utterance 標準化。

---

## 最新實驗 (2025-11-26)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=12, E100)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q12 \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q12 \
       --epochs 100 --batch-size 4 --lr 1e-4 --num-quantizer-layers 12\""
  ```

**架構與手法**:
- 與 N=4 / N=8 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2) 對前 N=12 個 RVQ 層做 code-level CE 對齊。
- 對每層 q∈{0…11} 個別算 CE(noisy_q→clean_q)，總 loss 為 12 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q12/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.2016 @ epoch 97**；Final Val Loss: 3.2020（略高於 N=8）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q12/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q12/inference_boy1_001_refined.wav \
    --num-quantizer-layers 12
  ```

**觀察 / 待辦**:
1. N=12 的 Val CE 反而高於 N=4/N=8，顯示對過多 RVQ 層進行 refinement 可能讓模型試圖修復資訊不足或極高頻細節，反而增加困難度。
2. 建議後續分析時聚焦 N=1 / N=4 / N=8 三種配置的聽感與 SI-SDR/PESQ/STOI；N=12 可視為「上限實驗」，用來確認多層 refinement 不會越多越好。
3. 若多層收益有限，未來方向可轉為：選擇性 refinement（修最重要的幾層）、加入 speaker/noise conditioning、以及小權重聲學 loss。 

---

## 最新實驗 (2025-12-01)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=12, E100, Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 指令（節錄）：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q12_officialmel \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 12 \
      --official-mel"
  ```

**架構與手法**:
- 與 baseline N=12 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 12 個 RVQ 層做 code-level CE 對齊。
- 透過 `--official-mel` 使用 MiMo 官方 wav2mel (config.nfft + log-mel) 產生 noisy/clean codes。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q12_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.3112 @ epoch 75**；Final Val Loss: 2.3131（原 N=12 baseline 約 3.2016，顯示官方前處理在大 N 下也有顯著幫助）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q12_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q12_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 12 \
    --official-mel
  ```

**觀察 / 待辦**:
1. 在 N=12 的設定下，official-mel 將 Val CE 從 ~3.20 拉低到 ~2.31，說明多層 RVQ 的困難度很大一部份來自前處理與 MiMo 預訓練不一致。
2. 雖然 N=12_officialmel 已經比舊版 N=12 好很多，但 CE 仍明顯高於 N=4/N=8 官方前處理版本；實務上仍較適合作為「上限實驗」而不是主線設定。
3. 建議後續主要聚焦 N=4/N=8 的官方前處理版本，加上 conditioning 與聲學 loss，再視需要選擇性地觀察大 N（N=12）行為。

---

## 最新實驗 (2025-11-28)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=12, E100, WaveNorm)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker（排程腳本的一部分，單獨列出指令如下）：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q12_norm \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 12 \
      --normalize-waveform"
  ```

**架構與手法**:
- 與 N=12 baseline 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 12 個 RVQ 層做 code-level CE 對齊。
- 差異點：啟用 per-utterance waveform normalization，noisy/clean waveform 在轉 Mel 前先做 `(x - mean) / std`，Mel 與 quantizer 結構維持不變。
- 資料流程：
  - normalized noisy/clean waveform → Mel → encoder.get_features → quantizer → `codes_noisy`, `codes_clean`（約 [20, T]）。
  - 使用前 12 個 RVQ 層（N=12），逐層 CE(noisy_q→clean_q)，總 loss 為 12 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q12_norm/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.3947 @ epoch 82**；Final Val Loss: 3.3953（高於未做 WaveNorm 的 N=12 實驗，後者約為 3.20）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q12_norm/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q12_norm/inference_boy1_001_refined.wav \
    --num-quantizer-layers 12
  ```

**觀察 / 待辦**:
1. N=12 WaveNorm 的 Val CE 約 3.39，明顯高於未正規化版本的 ~3.20，延續 N=4/N=8 的趨勢：在 CodeRefiner 任務下，per-utterance waveform normalization 一致造成 CE 上升。
2. 需搭配聽感比較 `code_refiner_optical_e100_q12` vs `code_refiner_optical_e100_q12_norm`，確認音質是否同樣退化；若無優勢，可將 WaveNorm 視為「負面對照」，主線仍維持原本不正規化設定。
3. 多層（N≥12）本身就偏難收斂，再疊加 WaveNorm 後，CE 進一步惡化，更強化「不要盲目擴大 N，而應聚焦在 N=1/N=4 + 更合理的條件訊號與聲學 loss」的結論。

---

## 最新實驗 (2025-11-26)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=16, E100)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q16 \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q16 \
       --epochs 100 --batch-size 4 --lr 1e-4 --num-quantizer-layers 16\""
  ```

**架構與手法**:
- 與 N=4 / 8 / 12 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2) 對前 N=16 個 RVQ 層做 code-level CE 對齊。
- 對每層 q∈{0…15} 個別算 CE(noisy_q→clean_q)，總 loss 為 16 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q16/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.4898 @ epoch 94**；Final Val Loss: 3.4901（高於 N=12）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q16/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q16/inference_boy1_001_refined.wav \
    --num-quantizer-layers 16
  ```

**觀察 / 待辦**:
1. N=16 的 Val CE 明顯高於 N=4/N=8，甚至高於 N=12，顯示對過多 RVQ 層進行 refinement 會增加學習難度且收益有限。
2. 多層實驗（N=4/8/12/16）整體趨勢：N=4 最佳、N=8 輕微退化、N=12/16 持續變差。建議後續聚焦 N=4（或 N=1+N=4 對照）作為 CodeRefiner 主線。
3. 未來方向：若要進一步提升，可優先嘗試選擇性 refinement（挑幾層關鍵 RVQ）、加入 speaker/noise conditioning，以及引入小權重聲學 loss，而不是繼續擴大 N。

---

## 最新實驗 (2025-12-01)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=16, E100, Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 指令：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q16_officialmel \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 16 \
      --official-mel"
  ```

**架構與手法**:
- 與 baseline N=16 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 16 個 RVQ 層做 code-level CE 對齊。
- 使用官方 wav2mel 前處理產生 codes，訓練 / 推理皆開啟 `--official-mel`。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q16_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **2.7074 @ epoch 74**；Final Val Loss: 2.7093（原 N=16 baseline 約 3.4898，official-mel 大幅改善）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q16_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q16_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 16 \
    --official-mel
  ```

**觀察 / 待辦**:
1. N=16_officialmel 的 Val CE 約 2.71，雖然仍高於 N=4/N=8/N=12 官方版本，但和舊 N=16 (≈3.49) 相比已有顯著進步，證實前處理對大 N 也同樣關鍵。
2. 由於 N=16 仍屬極難設定，仍建議只作為分析用上限實驗；主線仍以 N=4/N=8 官方前處理為主。
3. 若未來要在大 N 上做更多探索，建議搭配更強的模型容量或層別拆分，而非完全共享同一個 Refiner。

---

## 最新實驗 (2025-11-28)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=16, E100, WaveNorm)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 排程（節錄單次命令）：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q16_norm \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 16 \
      --normalize-waveform"
  ```

**架構與手法**:
- 與 N=16 baseline 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 16 個 RVQ 層做 code-level CE 對齊。
- 啟用 per-utterance waveform normalization，noisy/clean waveform 在進入 Mel 前正規化為 mean≈0、std≈1。
- 使用前 16 個 RVQ 層（N=16），逐層 CE(noisy_q→clean_q)，總 loss 為 16 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q16_norm/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.6028 @ epoch 97**；Final Val Loss: 3.6033（高於未做 WaveNorm 的 N=16 實驗，後者約為 3.49）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q16_norm/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q16_norm/inference_boy1_001_refined.wav \
    --num-quantizer-layers 16
  ```

**觀察 / 待辦**:
1. N=16 WaveNorm 的 Val CE ≈3.60，再次高於未正規化版本的 ≈3.49，持續延伸「WaveNorm 對 CodeRefiner CE 不利」的結論。
2. 由於 N=16 本身就比 N=12 更難學，再疊加 WaveNorm 後，loss 幾乎沒有改善空間，之後若要探討大 N 的行為，WaveNorm 可作為「負面對照」參考即可。
3. 建議後續僅保留少數代表性的大 N 實驗作為 ablation，實務主線仍集中在 N=1/N=4 上，並改把心力投到 conditioning + acoustic loss 上，而非 waveform normalization。

---

## 最新實驗 (2025-11-27)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=20, E100)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q20 \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q20 \
       --epochs 100 --batch-size 4 --lr 1e-4 --num-quantizer-layers 20\""
  ```

**架構與手法**:
- 與 N=4 / 8 / 12 / 16 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2) 對前 N=20 個 RVQ 層做 code-level CE 對齊。
- 對每層 q∈{0…19} 個別算 CE(noisy_q→clean_q)，總 loss 為 20 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q20/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.6608 @ epoch 100**；Final Val Loss: 3.6608（較 N=16 再略為變差）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q20/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q20/inference_boy1_001_refined.wav \
    --num-quantizer-layers 20
  ```

**觀察 / 待辦**:
1. N=20 的 Val CE 進一步高於 N=16，確認「越多 RVQ 層同時 refinement 並不會帶來收斂上的好處」，反而讓模型學習難度與不確定性同步增加。
2. 綜合 N=4/8/12/16/20 的實驗，可視 N=20 為「上限 ablation」，用來佐證多層 refinement 存在效益飽和甚至反向的情況；實務上較合理的工作點仍集中在 N=1/N=4。
3. 後續建議：停止再擴大 N，而是將重心轉向（1）選擇性 refinement（只修關鍵 RVQ 層）、（2）加入 speaker/noise conditioning、以及（3）在最佳 N 設定上疊加小權重聲學 loss 進一步微調。

---

## 最新實驗 (2025-12-01)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=20, E100, Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 指令：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q20_officialmel \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 20 \
      --official-mel"
  ```

**架構與手法**:
- 與 baseline N=20 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 20 個 RVQ 層做 code-level CE 對齊。
- 使用官方 wav2mel 前處理產生 noisy/clean codes，訓練／推理皆使用 `--official-mel`。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q20_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.0293 @ epoch 78**；Final Val Loss: 3.0300（原 N=20 baseline 約 3.6608，official-mel 有明顯改善，但仍高於 N≤16 官方版本）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q20_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q20_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 20 \
    --official-mel
  ```

**觀察 / 待辦**:
1. N=20_officialmel 的 Val CE 約 3.03，顯著優於舊 N=20 (≈3.66)，但仍明顯高於 N=4/8/12/16 官方版本，再次確認「N 越大越難」仍然存在，只是前處理不再是主要瓶頸。
2. 綜合所有官方前處理實驗，可將 N=1/4/8 視為較務實的工作點，大 N (12/16/20) 則多用於 ablation 與理解 RVQ 層次行為。
3. 之後若要進一步提升大 N，可考慮分層 Refiner 或增加模型容量，而不是完全依賴單一共享 Transformer。

---

## 最新實驗 (2025-12-02)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=20, E100, WaveNorm + Official Mel)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- tmux + docker：
  ```bash
  tmux new -s code_refiner_e100_q20_norm_officialmel \
    "docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \
     bash -lc \"python finetune_code_refiner.py \
       --data-dir ./data/splits/finetune_optical \
       --tokenizer-path ./models/MiMo-Audio-Tokenizer \
       --output-dir ./outputs/code_refiner_optical_e100_q20_norm_officialmel \
       --epochs 100 --batch-size 4 --lr 1e-4 \
       --num-quantizer-layers 20 \
       --normalize-waveform \
       --official-mel\""
  ```

**架構與手法**:
- 與 N=20_officialmel 相同：凍結 tokenizer，使用共享 Transformer Refiner 對前 20 個 RVQ 層做 code-level CE 對齊。
- 差異點：waveform 端啟用 per-utterance normalization，再接官方 wav2mel 前處理。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q20_norm_officialmel/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.4433 @ epoch 94**；Final Val Loss: 3.4440（略優於 N=20_officialmel ≈3.0293 的改善幅度有限，整體仍明顯高於 N≤16 的設定）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q20_norm_officialmel/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q20_norm_officialmel/inference_boy1_001_refined.wav \
    --num-quantizer-layers 20 \
    --official-mel
  ```

**觀察 / 待辦**:
1. 在 N=20 上，同時啟用 WaveNorm + official-mel 雖有小幅度 CE 改善，但 Val Loss 仍然遠高於 N=4/8/12/16 的官方前處理版本，顯示大 N 本身的難度仍是主要瓶頸。
2. 相較於 N=4_norm_officialmel 的明顯正面效果，N=20_norm_officialmel 僅能算是「微調」，實務上不建議把全部 20 層都丟給同一個 Refiner 處理。
3. 建議後續若要維持 N=20 的能力，可考慮改為分層 Refiner 或只精修前幾層 RVQ，讓後面層保留原始 codes，而不是試圖對所有層做強制對齊。

---

## 最新實驗 (2025-11-28)
### 🧪 CodeRefiner - 多層 RVQ Token Refinement (N=20, E100, WaveNorm)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 分支：`exp/code-refinement`
- 訓練腳本：`finetune_code_refiner.py`
- docker 排程（節錄）：
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace mimo-audio:latest \
    bash -lc "python finetune_code_refiner.py \
      --data-dir ./data/splits/finetune_optical \
      --tokenizer-path ./models/MiMo-Audio-Tokenizer \
      --output-dir ./outputs/code_refiner_optical_e100_q20_norm \
      --epochs 100 --batch-size 4 --lr 1e-4 \
      --num-quantizer-layers 20 \
      --normalize-waveform"
  ```

**架構與手法**:
- 與 N=20 baseline 相同：凍結 tokenizer，使用共享 Transformer Refiner (d_model=256, nhead=4, num_layers=2, ff=1024, dropout=0.1) 對前 20 個 RVQ 層做 code-level CE 對齊。
- 啟用 per-utterance waveform normalization，noisy/clean waveform 在進入 Mel 前先標準化。
- 使用前 20 個 RVQ 層（N=20），逐層 CE(noisy_q→clean_q)，總 loss 為 20 層 CE 的平均。

**關鍵結果**:
- 輸出目錄：`outputs/code_refiner_optical_e100_q20_norm/`（`best_model.pt`、每 10 epoch checkpoint、`training_history.json`、`inference_boy1_001_refined.wav`）
- 訓練指標：Epochs=100；Best Val Loss: **3.7324 @ epoch 97**；Final Val Loss: 3.7332（高於未做 WaveNorm 的 N=20 實驗，後者約為 3.66）
- 單檔推理：
  ```bash
  python test_code_refiner_inference.py \
    --checkpoint outputs/code_refiner_optical_e100_q20_norm/best_model.pt \
    --tokenizer-path models/MiMo-Audio-Tokenizer \
    --input examples/optical/mix/boy1_WOLDV_001.wav \
    --output outputs/code_refiner_optical_e100_q20_norm/inference_boy1_001_refined.wav \
    --num-quantizer-layers 20
  ```

**觀察 / 待辦**:
1. N=20 WaveNorm 的 Val CE ≈3.73，再次高於未正規化版本的 ≈3.66，多層（大 N）下的 WaveNorm 實驗全面呈現負面效果。
2. 綜合 N=4/8/12/16/20 的 WaveNorm 對照，可以比較明確地說：對 CodeRefiner 這種「noisy→clean code CE 對齊」任務，waveform 層的 per-utterance normalization 不但沒有穩定效果，反而削弱了模型利用振幅差異的能力。
3. 後續若要繼續探索 normalization，建議改在 Mel/feature 層重新設計（例如 per-band 標準化或 learnable norm），而不是直接對 waveform 做 global z-score；目前 CodeRefiner 主線實驗可維持在「不做 waveform normalize」的設定。

---

## 最新實驗 (2025-11-19)
### 🧪 Encoder Fine-tuning V2 - Waveform Normalization (pre-Mel) v2
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 訓練：`bash start_training_v2.sh`（`OUTPUT_DIR=outputs/optical_lora_v2_normbeforemel_r32_e100_v2`，預設啟用 `--normalize-waveform`）→ docker + tmux `finetune_v2`
- 推論：`CHECKPOINT=outputs/optical_lora_v2_normbeforemel_r32_e100_v2/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_normbeforemel_v2/enhanced_050.wav ./test_inference_docker.sh --no-tmux`

**關鍵結果**:
- Train Loss: 21.3 → 0.35；Val Loss: 7.78 → 0.33（最佳 0.33 @ epoch 100，仍有震盪）
- Checkpoints：`best_model.pt` + 每 10 epoch 快照；`training_history.json` 新增 100-epoch loss trace
- 推論樣本：`outputs/test_inference_normbeforemel_v2/enhanced_050.wav`（另存 `examples/codebook_align_inference/boy1_WOLDV_050_enhanced_v2_normbeforemel_v2.wav`）

**洞察 / 待辦**:
1. 仍可見多次梯度爆衝（Val loss >10），推測 batch variance 過大；下一輪嘗試降低 LR 或加入 grad clipping
2. waveform normalization 放在 Mel 前效果尚未顯著優於 11/18 版本，需客觀 SI-SDR/PESQ 檢查
3. 若要與先前 run 對照，保留 `outputs/optical_lora_v2_norm_r32_e100/` 與 `outputs/test_inference_v2_norm/`，勿覆蓋

---

## 過往實驗 (2025-11-18)
### 🧪 Encoder Fine-tuning V2 - Codebook Alignment + Waveform Normalization
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 訓練：`bash start_training_v2.sh`（啟用 `--normalize-waveform`，在轉 Mel 前將 waveform 正規化為 mean=0、std=1）→ docker + tmux `finetune_v2`，輸出 `outputs/optical_lora_v2_norm_r32_e100/`
- 推論：`CHECKPOINT=outputs/optical_lora_v2_norm_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2_norm/enhanced_050.wav ./test_inference_docker.sh --no-tmux`

**關鍵結果**:
- Train Loss: 6.06 → 0.063；Val Loss: 3.14 → 0.040（最佳 0.033 @ epoch 47）
- Waveform normalization（轉 Mel 前做 mean/std）讓 loss 尺度下降、收斂更快；未實作 per-band z-score
- Checkpoints：`best_model.pt` + 每 10 epoch 快照；新 log `training_history.json`
- 推論樣本：`outputs/test_inference_v2_norm/enhanced_050.wav`（另存 `examples/codebook_align_inference/boy1_WOLDV_050_enhanced_v2_norm.wav`）

**洞察 / 待辦**:
1. Waveform normalization 讓 loss 縮小、收斂更快，仍需配合 SI-SDR/PESQ 驗證主觀音質是否同步提升
2. 建議使用同一 script 批次產線：`CHECKPOINT=outputs/optical_lora_v2_norm_r32_e100/best_model.pt bash test_inference_quick.sh`
3. 下一步：修補 splits＋加入 early stopping，再重訓比較 waveform norm vs baseline 表現
4. 若需回溯舊結果（Mel normalization），保留 `outputs/test_inference_v2/enhanced_050.wav` 以對照

---

## 過往實驗 (2025-11-17)
### 🧪 Encoder Fine-tuning V2 - Codebook Alignment (LoRA Rank 32)
**狀態**: ✅ 已完成（100/100 epochs + 單檔 inference）

**執行方式**:
- 訓練：`bash start_training_v2.sh` → docker + tmux，輸出於 `outputs/optical_lora_v2_r32_e100/`
- 推論：`CHECKPOINT=outputs/optical_lora_v2_r32_e100/best_model.pt INPUT=examples/optical/mix/boy1_WOLDV_050.wav OUTPUT=outputs/test_inference_v2/enhanced_050.wav ./test_inference_docker.sh --no-tmux`

**關鍵結果**:
- Loss 組成：Feature MSE + Codebook L1 + VQ commit（`lambda_feat=1, lambda_code=1, lambda_vq=0.1`）
- Train Loss: 12.08 → 8.22；Val Loss: 9.23 → 8.20（最佳 8.06 @ epoch 2）
- Checkpoints：`best_model.pt` + 每 10 epoch 快照；訓練曲線 `training_history.json`
- 推論樣本：`outputs/test_inference_v2/enhanced_050.wav`（與 `examples/optical/mix/boy1_WOLDV_050.wav` 對照）

**洞察 / 待辦**:
1. Codebook-aware Loss 可穩定訓練，但仍需檢查資料洩漏對 Val 的影響
2. 建議以 `test_inference_docker.sh` 批次生成樣本並計算 SI-SDR/PESQ
3. 若要縮短訓練，可加入 early stopping（最佳點依舊落在前幾個 epoch）

---

## 過往實驗 (2025-11-13)
### 🧪 Encoder Fine-tuning - LoRA Rank 32, 100 Epochs
**狀態**: ✅ 已完成  
**詳細報告**: [FINETUNE_EXPERIMENT_R32_E100.md](FINETUNE_EXPERIMENT_R32_E100.md)

**核心發現**:
- ✅ Training 成功完成 100 epochs (33.51% loss 改善)
- ✅ 無過擬合 (Train-Val gap 僅 0.28%)
- ⚠️ 最佳 Val loss 在 Epoch 2 出現，100 epochs 過多
- ⚠️ 發現資料集洩漏問題 (196/136/208 樣本重疊)

**後續行動**:
1. 修復資料洩漏，重新生成 splits
2. 用乾淨資料集重訓，加入 early stopping
3. 客觀評估音訊品質 (SI-SDR, PESQ)

---

## Docker 實驗 (2025-10-13)

## 實驗目的
驗證 MiMo-Audio 模型在 Docker 容器中的完整功能，包括：
1. SFT (Supervised Fine-Tuning) 模型的多種任務能力
2. Base 預訓練模型的 In-Context Learning 能力

## 環境配置

### Docker 環境
- **Base Image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.12
- **PyTorch**: 2.6.0
- **TorchAudio**: 2.6.0
- **CUDA**: 12.1
- **Flash Attention**: 2.7.4.post1

### 硬體需求
- GPU with CUDA support
- 至少 16GB VRAM（用於 7B 模型）
- 約 40GB 磁碟空間（包含兩個模型）

## 實驗一：SFT 模型測試

### 模型
- **模型名稱**: MiMo-Audio-7B-Instruct
- **模型大小**: ~15GB
- **模型類型**: 指令微調模型

### 測試腳本
`inference_example_sft.py`

### 測試功能

#### 1. 基本 TTS (Text-to-Speech)
- **輸入文本**: "今天天气真好"
- **輸出檔案**: `examples/tts.wav` (233 KB)
- **結果**: ✅ 成功生成自然流暢的中文語音

#### 2. 指令式 TTS
- **輸入文本**: "今天天气真好"
- **指令**: "用小孩子的声音开心的说"
- **輸出檔案**: `examples/instruct_tts.wav` (106 KB)
- **結果**: ✅ 成功模擬小孩子的聲音特徵和情緒

#### 3. 自然指令 TTS
- **輸入文本**: "用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！"
- **輸出檔案**: `examples/natural_instruction_tts.wav` (106 KB)
- **結果**: ✅ 成功表現氣喘吁吁的效果

#### 4. 音頻理解
- **輸入音頻**: `examples/spoken_dialogue_assistant_turn_1.wav`
- **任務**: "Summarize the audio."
- **結果**: ✅ 成功總結音頻內容（英文摘要）

#### 5. 音頻理解（帶思考過程）
- **輸入音頻**: `examples/spoken_dialogue_assistant_turn_1.wav`
- **任務**: "Summarize the audio."
- **結果**: ✅ 提供詳細的思考過程和分析

#### 6. 多輪口語對話
- **輸入**: 
  - 第一輪：音頻問天氣
  - 第二輪：回答「北京」
- **輸出檔案**: `examples/spoken_dialogue_assistant_turn_2.wav` (1.4 MB)
- **結果**: ✅ 成功生成連貫的多輪對話音頻

#### 7. 語音轉文字對話
- **輸入**: 多輪語音和文字混合對話
- **結果**: ✅ 成功處理語音輸入並生成文字回應

#### 8. 純文字對話
- **任務**: 介紹北京的旅遊景點
- **結果**: ✅ 生成詳細的旅遊推薦（包含景點、交通、門票等資訊）

### 性能指標
- **模型載入時間**: ~8 秒
- **Tokenizer 載入時間**: ~11 秒
- **單次推理時間**: 視任務複雜度而定（2-30 秒）

## 實驗二：Base 模型 In-Context Learning 測試

### 模型
- **模型名稱**: MiMo-Audio-7B-Base
- **模型大小**: ~15GB
- **模型類型**: 預訓練基礎模型

### 測試腳本
`inference_example_pretrain.py`

### 測試任務：音色轉換 (Voice Conversion)

#### 實驗設計
- **任務**: 將說話人 0013 的聲音轉換為說話人 0019 的音色
- **方法**: In-Context Learning (Few-shot Learning)
- **示例數量**: 5 對語音轉換示例

#### 示例對（說話人 0013 → 0019）
1. `0013_000139.wav` → `0019_000139.wav`: "Cuckoos is downheaded and crying."
2. `0013_000963.wav` → `0019_000963.wav`: "She said in subdued voice."
3. `0013_000559.wav` → `0019_000559.wav`: "A raging fire was-in his eyes."
4. `0013_001142.wav` → `0019_001142.wav`: "Does the one that wins get the crowned?"
5. `0013_000769.wav` → `0019_000769.wav`: "Not much use is it, sam?"

#### 測試輸入
- **輸入音頻**: `examples/ESD/0013_000200.wav` (說話人 0013)
- **指令**: "Convert the timbre of the input speech to target timbre."

#### 測試結果
- **輸出檔案**: `examples/in_context_learning_s2s.wav` (136 KB)
- **轉錄文本**: "The cloth does not look worth much."
- **結果**: ✅ 成功將音色從說話人 0013 轉換為說話人 0019

### 性能指標
- **模型載入時間**: ~8 秒
- **Tokenizer 載入時間**: ~10 秒
- **推理時間**: 包含處理 5 個示例對 + 生成輸出
- **max_new_tokens**: 8192

## 實驗結論

### 成功驗證的功能
1. ✅ Docker 環境配置正確，GPU 加速正常
2. ✅ PyTorch 2.6.0 與 CUDA 12.1 相容性良好
3. ✅ SFT 模型支援多種任務（TTS、對話、理解）
4. ✅ Base 模型支援 In-Context Learning
5. ✅ 模型載入和推理穩定可靠
6. ✅ 音頻生成質量良好

### 兩種模型的差異
| 特性 | Base (Pretrain) | Instruct (SFT) |
|------|-----------------|----------------|
| **使用方式** | Few-shot examples | 直接指令 |
| **靈活性** | 高（可學習新任務） | 中（預定義任務） |
| **易用性** | 低（需準備示例） | 高（直接調用） |
| **適用場景** | 研究探索 | 生產應用 |

## 如何重現實驗

### 前置準備

#### 1. 下載模型
```bash
# 下載 Instruct 模型（用於實驗一）
huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --local-dir models/MiMo-Audio-7B-Instruct

# 下載 Base 模型（用於實驗二）
huggingface-cli download XiaomiMiMo/MiMo-Audio-7B-Base \
  --local-dir models/MiMo-Audio-7B-Base

# 下載 Tokenizer
huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir models/MiMo-Audio-Tokenizer
```

#### 2. 構建 Docker 映像
```bash
docker build -t mimo-audio:latest .
```

### 重現實驗一：SFT 模型測試

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/examples:/app/examples \
  mimo-audio:latest \
  python3.12 inference_example_sft.py
```

**預期輸出檔案**:
- `examples/tts.wav`
- `examples/instruct_tts.wav`
- `examples/natural_instruction_tts.wav`
- `examples/spoken_dialogue_assistant_turn_2.wav`

### 重現實驗二：Base 模型 In-Context Learning

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/examples:/app/examples \
  mimo-audio:latest \
  python3.12 inference_example_pretrain.py
```

**預期輸出檔案**:
- `examples/in_context_learning_s2s.wav`

### 快速測試（使用 smoke test）

```bash
docker run --gpus all --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  mimo-audio:latest \
  python3.12 scripts/smoke_test.py \
  --audio ./examples/prompt_speech_zh_m.wav
```

## 生成的檔案清單

### 實驗一生成的檔案
- `examples/tts.wav` - 基本 TTS 輸出
- `examples/instruct_tts.wav` - 指令式 TTS（小孩子聲音）
- `examples/natural_instruction_tts.wav` - 自然指令 TTS（氣喘吁吁）
- `examples/spoken_dialogue_assistant_turn_2.wav` - 多輪對話第二輪回應

### 實驗二生成的檔案
- `examples/in_context_learning_s2s.wav` - 音色轉換結果

### Smoke Test 生成的檔案
- `outputs/tts_sample.wav` - 測試語音樣本

## 問題排查

### 常見問題

**Q1: 遇到 CUDA out of memory 錯誤**
```
A: 需要至少 16GB VRAM。可以嘗試：
   - 減少 batch size
   - 使用更小的模型
   - 關閉其他 GPU 程序
```

**Q2: 模型載入時間過長**
```
A: 正常現象。7B 模型需要 7-10 秒載入時間。
   可以使用模型緩存加速後續載入。
```

**Q3: 音頻質量不佳**
```
A: 檢查：
   - 輸入音頻質量
   - 採樣率是否正確
   - 是否使用了正確的模型
```

## 技術細節

### Docker 映像層級
1. NVIDIA CUDA 12.1.1 runtime base
2. Python 3.12 安裝
3. PyTorch 2.6.0 + TorchAudio 2.6.0
4. Flash Attention 2.7.4
5. 專案依賴安裝

### 關鍵依賴版本
- `torch==2.6.0`
- `torchaudio==2.6.0`
- `transformers==4.49.0`
- `accelerate>=1.9.0`
- `librosa>=0.11.0`
- `gradio==5.46.1`

## 未來改進方向

1. **性能優化**
   - 實現模型量化（INT8/INT4）
   - 支援批次推理
   - 優化記憶體使用

2. **功能擴展**
   - 支援更多語言
   - 增加實時推理能力
   - Web UI 整合

3. **測試擴展**
   - 增加自動化測試
   - 音頻品質評估指標
   - 效能基準測試

## 參考資料
- MiMo-Audio GitHub: https://github.com/XiaomiMiMo/MiMo-Audio
- Hugging Face Models: https://huggingface.co/XiaomiMiMo

## 實驗三：optical 正規化後的初步降噪（noisereduce）

### 目的
- `examples/optical_normalized/{mix,spk1}` 仍可聽見高頻砂點/顆粒噪聲；以 noisereduce 先做保守降噪，供 few‑shot 與主觀評測使用。

### 流程與參數
- 腳本：`scripts/denoise_optical.py`
- 來源與輸出：
  - 來源：`examples/optical_normalized/{mix,spk1}`
  - 輸出：`examples/optical_denoised/{mix,spk1}`（mirror 同名）
- 前處理濾波：`highpass=90 Hz`、`lowpass=8500 Hz`
- 降噪：noisereduce 非平穩（`stationary=False`）、`prop_decrease=0.85`
- 平滑：`time_mask_smooth_ms=64`、`freq_mask_smooth_hz=150`
- 後處理：峰值正規化 `-1.0 dBFS`，輸出 `WAV PCM16`

### 執行（Docker）
```bash
docker build -t mimo-audio:latest .
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/app -w /app mimo-audio:latest \
  python -u scripts/denoise_optical.py
```

### 結果摘要（2025-11-06）
- 檔案數：
  - mix：3456 個
  - spk1：3456 個
- 容量：
  - `examples/optical_denoised/mix`：435 MB
  - `examples/optical_denoised/spk1`：435 MB
- 備註：輸出檔依 `.gitignore` 規則忽略版本控管。
