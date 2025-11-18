#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune MiMo-Audio Encoder with LoRA - Version 2
改進策略：Codebook-Aware Training

核心思想：
1. 保持 Codebook + Decoder 不變（利用小米模型強大能力）
2. 訓練 Encoder 將 noisy audio 映射到與 clean audio 相同的 codebook
3. 使用 Codebook Alignment Loss 確保一致性
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import torchaudio
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.mimo_audio_tokenizer import MiMoAudioTokenizer


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) Layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 低秩分解: W = W_0 + BA
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        # ΔW = BA * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora_to_encoder(encoder, rank=32, alpha=64):
    """
    將 LoRA 注入到 Encoder 的所有 Linear layers
    保持原始權重凍結，只訓練 LoRA 參數
    """
    lora_count = 0
    
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear):
            # 凍結原始權重
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            # 注入 LoRA
            in_features = module.in_features
            out_features = module.out_features
            lora = LoRALayer(in_features, out_features, rank, alpha)
            # keep LoRA tensors on the same device as the frozen linear layer
            lora = lora.to(module.weight.device)
            
            # Monkey-patch forward
            original_forward = module.forward
            def new_forward(x, lora_layer=lora, orig_forward=original_forward):
                return orig_forward(x) + lora_layer(x)
            module.forward = new_forward
            
            # 註冊 LoRA 參數
            setattr(module, 'lora', lora)
            lora_count += 1
    
    logger.info(f"✅ Injected LoRA to {lora_count} Linear layers")
    
    # 統計可訓練參數
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"📊 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return encoder


def compute_codebook_aware_loss(
    encoder,
    noisy_audio,
    clean_audio,
    device,
    lambda_feat=1.0,
    lambda_code=1.0,
    lambda_vq=0.1,
):
    """
    改進的損失函數：Codebook-Aware Loss
    
    核心思想：
    1. Feature matching loss: 特徵空間對齊
    2. Codebook alignment loss: 確保映射到相同的 codebook codes（使用 quantizer forward，可回傳梯度）
    3. Code consistency loss: 透過 quantizer 的 commit loss 維持碼本穩定

    Args:
        encoder: Audio Encoder (with LoRA)
        noisy_audio: [batch, n_mels, time] mel spectrogram
        clean_audio: [batch, n_mels, time] mel spectrogram
        lambda_feat: feature loss 權重
        lambda_code: codebook loss 權重
    """
    # noisy_audio/clean_audio 已經是 mel spectrogram [batch, n_mels, time]
    batch_size = noisy_audio.shape[0]
    mel_len = noisy_audio.shape[2]
    mel_lens = torch.tensor([mel_len] * batch_size, device=device, dtype=torch.long)
    output_length = encoder.get_output_length(mel_lens)
    
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # === 1. Feature Matching Loss ===
        noisy_features = encoder.get_features(
            input_features=noisy_audio.to(torch.bfloat16),
            output_length=output_length
        )[0]
        
        with torch.no_grad():
            clean_features = encoder.get_features(
                input_features=clean_audio.to(torch.bfloat16),
                output_length=output_length
            )[0]
        
        feature_loss = F.mse_loss(noisy_features, clean_features)
        
        # === 2. Codebook Alignment Loss ===
        # 使用 quantizer forward 取得可微分的 codebook 表示
        quantizer = getattr(encoder, "quantizer", None)
        if quantizer is None:
            raise ValueError("Encoder does not expose a quantizer; cannot run codebook-aware loss.")
        
        # quantizer 可能沒有參數（純 buffer 代碼本），因此需安全取得 dtype
        try:
            quantizer_dtype = next(quantizer.parameters()).dtype
        except StopIteration:
            quantizer_buffer = next(quantizer.buffers(), None)
            quantizer_dtype = quantizer_buffer.dtype if quantizer_buffer is not None else noisy_features.dtype
        noisy_flat = noisy_features.reshape(-1, noisy_features.shape[-1]).to(quantizer_dtype)
        quantized_noisy, _, vq_commit_loss, _ = quantizer(noisy_flat)
        quantized_noisy = quantized_noisy.view_as(noisy_features)
        
        with torch.no_grad():
            clean_flat = clean_features.reshape(-1, clean_features.shape[-1]).to(quantizer_dtype)
            quantized_clean, _, _, _ = quantizer(clean_flat)
            quantized_clean = quantized_clean.view_as(clean_features)
        
        code_loss = F.l1_loss(quantized_noisy, quantized_clean)
        
        # === 3. Perceptual Loss (optional) ===
        # 使用不同層的特徵（如果有 skip connection）
        # 這裡暫時省略，可以後續加入
        
        # 總損失
        total_loss = (
            lambda_feat * feature_loss
            + lambda_code * code_loss
            + lambda_vq * vq_commit_loss
        )
    
    return total_loss, feature_loss, code_loss


class OpticalDataset(Dataset):
    """Optical 噪聲音訊配對資料集

    重點：
    - 直接在 Dataset 裡把 waveform 轉成固定長度的 Mel Spectrogram
    - 確保輸出形狀為 [n_mels, time]，方便 DataLoader batch 起來
    - Mel 參數需與 MiMo-Audio-Tokenizer 的 config 一致
    """

    def __init__(
        self,
        data_dir,
        split='train',
        max_length=10.0,
        sample_rate=24000,
        normalize=False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length  # 以秒為單位
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)
        self.normalize = normalize
        self.eps = 1e-6

        # Mel Spectrogram 轉換器（與 MiMo-Audio-Tokenizer 一致）
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=240,
            n_mels=128,
            f_min=0.0,
            f_max=float(sample_rate // 2),
        )

        # 讀取 manifest
        manifest_file = self.data_dir / f"{split}.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            # 資料格式: {"samples": [...]}
            self.data = manifest['samples'] if 'samples' in manifest else manifest

        if not self.data:
            raise ValueError(f"No samples found in manifest: {manifest_file}")

        logger.info(f"📦 Loaded {len(self.data)} samples from {split} split")

    def __len__(self):
        return len(self.data)

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """載入單一音檔並轉成固定長度的 Mel Spectrogram [n_mels, time]."""
        waveform, sr = torchaudio.load(path)

        # 重新採樣到 target sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 轉 mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 截斷或 padding 到固定長度 self.max_samples
        num_samples = waveform.shape[1]
        if num_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif num_samples < self.max_samples:
            pad_len = self.max_samples - num_samples
            waveform = F.pad(waveform, (0, pad_len), value=0.0)

        # 轉成 mel spectrogram: [1, n_mels, time] -> [n_mels, time]
        mel_spec = self.mel_transform(waveform).squeeze(0)
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean(dim=-1, keepdim=True)) / (
                mel_spec.std(dim=-1, keepdim=True).clamp_min(self.eps)
            )
        return mel_spec

    def __getitem__(self, idx):
        item = self.data[idx]

        noisy_path = str(Path(item['noisy_file']))
        clean_path = str(Path(item['clean_file']))

        try:
            noisy_mel = self._load_and_preprocess(noisy_path)
            clean_mel = self._load_and_preprocess(clean_path)
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({item.get('id', idx)}): {e}")
            # 萬一有問題，回傳全零張量避免 DataLoader 崩潰
            mel_time = (self.max_samples // 240) + 1  # hop_length = 240
            noisy_mel = torch.zeros(128, mel_time)
            clean_mel = torch.zeros(128, mel_time)

        return {
            'noisy_audio': noisy_mel,   # [n_mels, time]
            'clean_audio': clean_mel,   # [n_mels, time]
            'filename': item.get('id', f'{idx}'),
        }


def train_one_epoch(
    encoder,
    dataloader,
    optimizer,
    device,
    epoch,
    gradient_accumulation_steps=1,
    lambda_code=1.0,
    lambda_vq=0.1,
):
    encoder.train()
    total_loss = 0.0
    total_feat_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        # 計算損失
        loss, feat_loss, code_loss = compute_codebook_aware_loss(
            encoder,
            noisy_audio,
            clean_audio,
            device,
            lambda_feat=1.0,
            lambda_code=lambda_code,
            lambda_vq=lambda_vq,
        )
        
        # Gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_feat_loss += feat_loss.item()
        total_code_loss += code_loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'feat': f'{feat_loss.item():.4f}',
            'code': f'{code_loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_feat = total_feat_loss / num_batches
    avg_code = total_code_loss / num_batches
    
    return avg_loss, avg_feat, avg_code


@torch.no_grad()
def validate(encoder, dataloader, device, lambda_code=1.0, lambda_vq=0.1):
    encoder.eval()
    total_loss = 0.0
    total_feat_loss = 0.0
    total_code_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        loss, feat_loss, code_loss = compute_codebook_aware_loss(
            encoder,
            noisy_audio,
            clean_audio,
            device,
            lambda_feat=1.0,
            lambda_code=lambda_code,
            lambda_vq=lambda_vq,
        )
        
        total_loss += loss.item()
        total_feat_loss += feat_loss.item()
        total_code_loss += code_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_feat_loss / num_batches, total_code_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/splits/finetune_optical')
    parser.add_argument('--tokenizer-path', type=str, default='./models/MiMo-Audio-Tokenizer')
    parser.add_argument('--output-dir', type=str, default='./outputs/optical_lora_v2_r32')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--lambda-code', type=float, default=1.0, help='Weight for codebook alignment loss')
    parser.add_argument('--lambda-vq', type=float, default=0.1, help='Weight for quantizer commit loss')
    parser.add_argument('--normalize-mel', action='store_true', help='Apply per-band mean/std normalization to mel spectrogram inputs')
    
    args = parser.parse_args()
    
    # 設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 Fine-tuning MiMo-Audio Encoder (Codebook-Aware)")
    logger.info("=" * 80)
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"🎯 LoRA: rank={args.rank}, alpha={args.alpha}")
    logger.info(f"📦 Batch: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    logger.info(f"📚 Epochs: {args.epochs}")
    logger.info(f"📈 LR: {args.lr}")
    
    # 載入模型
    logger.info("\n1️⃣ Loading MiMo-Audio-Tokenizer...")
    tokenizer = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer = tokenizer.to(device).to(torch.bfloat16)
    
    # 注入 LoRA
    logger.info("\n2️⃣ Injecting LoRA to Encoder...")
    inject_lora_to_encoder(tokenizer.encoder, rank=args.rank, alpha=args.alpha)
    
    # 準備資料
    logger.info("\n3️⃣ Loading Dataset...")
    train_dataset = OpticalDataset(
        args.data_dir,
        split='train',
        normalize=args.normalize_mel,
    )
    val_dataset = OpticalDataset(
        args.data_dir,
        split='val',
        normalize=args.normalize_mel,
    )

    if args.normalize_mel:
        logger.info("🎛️  Enabled per-band mel normalization for datasets")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 優化器
    optimizer = torch.optim.AdamW(
        tokenizer.encoder.parameters(),
        lr=args.lr, weight_decay=0.01
    )
    
    # 訓練
    logger.info("\n4️⃣ Starting Training...")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_feat': [], 'train_code': []}
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_feat, train_code = train_one_epoch(
            tokenizer.encoder,
            train_loader,
            optimizer,
            device,
            epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
            lambda_code=args.lambda_code,
            lambda_vq=args.lambda_vq,
        )
        
        val_loss, val_feat, val_code = validate(
            tokenizer.encoder,
            val_loader,
            device,
            lambda_code=args.lambda_code,
            lambda_vq=args.lambda_vq,
        )
        
        logger.info(f"📊 Train Loss: {train_loss:.4f} (feat={train_feat:.4f}, code={train_code:.4f})")
        logger.info(f"📊 Val Loss:   {val_loss:.4f} (feat={val_feat:.4f}, code={val_code:.4f})")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_feat'].append(train_feat)
        history['train_code'].append(train_code)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'lora_state_dict': {
                    name: param.data
                    for name, param in tokenizer.encoder.named_parameters()
                    if param.requires_grad
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'rank': args.rank,
                    'alpha': args.alpha,
                }
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"💾 Saved best model (val_loss={val_loss:.4f})")
        
        # 定期保存
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # 保存訓練歷史
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Completed!")
    logger.info(f"📊 Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"💾 Model saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
