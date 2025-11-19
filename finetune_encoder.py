#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune MiMo-Audio Encoder with LoRA"""

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


class OpticalDataset(Dataset):
    def __init__(self, split_file: str, max_duration: float = 10.0, sample_rate: int = 24000):
        self.split_file = Path(split_file)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        
        # 建立 Mel Spectrogram 轉換器（與 MiMo-Audio-Tokenizer 配置一致）
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=240,
            n_mels=128,
            f_min=0.0,
            f_max=float(sample_rate // 2)
        )
        
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(self.split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        self.samples = split_data.get('samples', [])
        if not self.samples:
            raise ValueError(f"No samples found in {split_file}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.split_file.name}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_and_preprocess_audio(self, path: str) -> torch.Tensor:
        """載入音訊並轉換為 mel spectrogram"""
        waveform, sr = torchaudio.load(path)
        
        # 重新採樣到 24kHz
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 轉換為 mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 截斷或填充到固定長度
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_length = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_length), value=0.0)
        
        # 轉換為 mel spectrogram: [1, n_mels, time]
        mel_spec = self.mel_transform(waveform)
        
        return mel_spec.squeeze(0)  # [n_mels, time]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            noisy_audio = self.load_and_preprocess_audio(sample['noisy_file'])
            clean_audio = self.load_and_preprocess_audio(sample['clean_file'])
            
            return {
                'noisy_audio': noisy_audio,
                'clean_audio': clean_audio,
                'sample_id': sample.get('id', str(idx)),
                'transcription': sample.get('transcription', '')
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # 返回與 mel spectrogram 形狀一致的零張量 [n_mels, time]
            mel_time = (self.max_length // 240) + 1  # hop_length = 240
            return {
                'noisy_audio': torch.zeros(128, mel_time),
                'clean_audio': torch.zeros(128, mel_time),
                'sample_id': f'error_{idx}',
                'transcription': ''
            }


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_to_encoder(encoder: nn.Module, rank: int = 8, alpha: float = 16.0):
    lora_modules = {}
    
    # First, collect all modules that need LoRA (convert to list to avoid iterator issues)
    modules_to_modify = []
    for name, module in list(encoder.named_modules()):
        if isinstance(module, nn.Linear) and any(keyword in name for keyword in ['attn', 'fc1', 'fc2', 'mlp']):
            modules_to_modify.append((name, module))
    
    logger.info(f"Found {len(modules_to_modify)} linear layers to inject LoRA")
    
    # Now modify them
    for name, module in modules_to_modify:
        lora = LoRALayer(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha
        )
        
        # 將 LoRA 層移到與原始模組相同的 device
        lora = lora.to(module.weight.device)
        
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        
        # Store lora module reference
        lora_modules[name] = lora
        
        # Monkey-patch the forward method
        original_forward = module.forward
        
        def make_forward_with_lora(orig_forward, lora_layer):
            def forward_with_lora(x):
                base_output = orig_forward(x)
                lora_output = lora_layer(x)
                return base_output + lora_output
            return forward_with_lora
        
        module.forward = make_forward_with_lora(original_forward, lora)
        
        # Register LoRA parameters directly to the module
        module.lora_A = lora.lora_A
        module.lora_B = lora.lora_B
    
    logger.info(f"Injected LoRA to {len(lora_modules)} linear layers (rank={rank}, alpha={alpha})")
    
    return lora_modules


def compute_feature_matching_loss(encoder: nn.Module, noisy_mel: torch.Tensor, clean_mel: torch.Tensor, device) -> torch.Tensor:
    """
    計算特徵匹配損失（簡化版本）
    目標：讓 encoder 從噪音音訊中提取的特徵接近乾淨音訊的特徵
    
    Args:
        encoder: Audio Encoder
        noisy_mel: [batch, n_mels, time] 噪音 mel spectrogram
        clean_mel: [batch, n_mels, time] 乾淨 mel spectrogram
    """
    batch_size = noisy_mel.shape[0]
    mel_len = noisy_mel.shape[2]
    
    # 計算 mel 長度
    mel_lens = torch.tensor([mel_len] * batch_size, device=device, dtype=torch.long)
    
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # 編碼噪音音訊（允許梯度）- 轉換為 bfloat16
        noisy_features = encoder.get_features(
            input_features=noisy_mel.to(torch.bfloat16),  # [batch, n_mels, time]
            output_length=encoder.get_output_length(mel_lens)
        )[0]  # 只取 hidden_states
        
        # 編碼乾淨音訊（作為目標，不需要梯度）
        with torch.no_grad():
            clean_features = encoder.get_features(
                input_features=clean_mel.to(torch.bfloat16),
                output_length=encoder.get_output_length(mel_lens)
            )[0]
        
        # 計算特徵空間的 MSE 損失
        loss = F.mse_loss(noisy_features, clean_features)
    
    return loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        loss = compute_feature_matching_loss(model.encoder, noisy_audio, clean_audio, device)
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        loss = compute_feature_matching_loss(model.encoder, noisy_audio, clean_audio, device)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model: {best_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MiMo-Audio Encoder with LoRA')
    
    parser.add_argument('--train-split', type=str, required=True, help='Training split JSON file')
    parser.add_argument('--val-split', type=str, required=True, help='Validation split JSON file')
    parser.add_argument('--tokenizer-path', type=str, default='./models/MiMo-Audio-Tokenizer', help='Path to MiMo-Audio-Tokenizer')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output-dir', type=str, default='./outputs/finetune_encoder', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    split_hash_path = Path(args.train_split).parent / 'split_hashes.json'
    if split_hash_path.exists():
        try:
            with open(split_hash_path, 'r', encoding='utf-8') as f:
                split_meta = json.load(f)
            logger.info("📂 Using dataset split:")
            logger.info(f"   • manifest : {split_meta.get('manifest')}")
            logger.info(f"   • seed      : {split_meta.get('seed')}")
            logger.info(f"   • generated : {split_meta.get('generated_at')}")
            for split_name, payload in split_meta.get('splits', {}).items():
                logger.info(f"   • {split_name:5s}: total={payload.get('total')} hash={payload.get('combined_hash')}")
        except Exception as err:
            logger.warning(f"⚠️  Failed to read split hash file at {split_hash_path}: {err}")
    
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer = tokenizer.to(device)
    
    # 將模型轉換為 bfloat16 以支援 Flash Attention
    tokenizer = tokenizer.to(torch.bfloat16)
    
    logger.info("Injecting LoRA to encoder...")
    lora_modules = inject_lora_to_encoder(tokenizer.encoder, rank=args.lora_rank, alpha=args.lora_alpha)
    
    trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in tokenizer.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    logger.info("Loading datasets...")
    train_dataset = OpticalDataset(args.train_split)
    val_dataset = OpticalDataset(args.val_split)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, tokenizer.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        train_loss = train_one_epoch(tokenizer, train_loader, optimizer, device, epoch, args.gradient_accumulation_steps)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        val_loss = validate(tokenizer, val_loader, device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(tokenizer, optimizer, epoch, val_loss, output_dir, is_best=is_best)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
