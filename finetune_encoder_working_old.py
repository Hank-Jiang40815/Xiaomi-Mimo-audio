#!/usr/bin/env python3
"""
MiMo-Audio Encoder 微調腳本 - 完整實現版本
針對嚴重噪聲場景微調 Audio Tokenizer 的 Encoder

使用 LoRA (Low-Rank Adaptation) 進行高效微調
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
from datetime import datetime

# 設置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from src.mimo_audio_tokenizer import MiMoAudioTokenizer


class AudioPairDataset(Dataset):
    """音訊配對資料集"""
    
    def __init__(self, data_dir, split='train', max_length=16000*10):  # 10秒
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        
        # 載入 split 檔案
        split_file = self.data_dir / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        if 'samples' in split_data:
            self.samples = split_data['samples']
        else:
            raise ValueError("Invalid split format")
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def load_audio(self, path):
        """載入音訊檔案"""
        waveform, sr = torchaudio.load(path)
        
        # 轉換到 24kHz（MiMo-Audio 的採樣率）
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)
        
        # 轉換為 mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 截斷或填充到固定長度
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        return waveform.squeeze(0)  # [T]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            noisy_audio = self.load_audio(sample['noisy_file'])
            clean_audio = self.load_audio(sample['clean_file'])
            
            return {
                'noisy_audio': noisy_audio,
                'clean_audio': clean_audio,
                'id': sample.get('id', str(idx))
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # 返回零張量避免中斷訓練
            return {
                'noisy_audio': torch.zeros(self.max_length),
                'clean_audio': torch.zeros(self.max_length),
                'id': f'error_{idx}'
            }


def apply_lora_to_encoder(encoder, rank=8, alpha=16):
    """為 Encoder 的線性層添加 LoRA"""
    
    lora_layers = []
    
    for name, module in encoder.named_modules():
        # 只對 Transformer 層中的線性層添加 LoRA
        if isinstance(module, nn.Linear) and ('attn' in name or 'mlp' in name):
            # 獲取原始權重的維度
            in_features = module.in_features
            out_features = module.out_features
            
            # 創建 LoRA 矩陣
            lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            lora_B = nn.Parameter(torch.zeros(out_features, rank))
            
            # 註冊到模組
            module.register_parameter(f'lora_A', lora_A)
            module.register_parameter(f'lora_B', lora_B)
            module.lora_alpha = alpha
            module.lora_rank = rank
            
            # 凍結原始權重
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            # 修改 forward 方法以包含 LoRA
            original_forward = module.forward
            
            def forward_with_lora(self, x):
                result = original_forward(x)
                lora_result = (x @ self.lora_A.T @ self.lora_B.T) * (self.lora_alpha / self.lora_rank)
                return result + lora_result
            
            module.forward = forward_with_lora.__get__(module, nn.Linear)
            lora_layers.append(name)
    
    logger.info(f"Applied LoRA to {len(lora_layers)} layers")
    return lora_layers


def train_epoch(model, dataloader, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        # 計算音訊長度
        batch_size = noisy_audio.shape[0]
        input_lens = torch.tensor([noisy_audio.shape[1]] * batch_size, device=device)
        
        # 編碼噪聲音訊
        noisy_features = model.encode(noisy_audio.unsqueeze(1), input_lens=input_lens, use_quantizer=False)
        
        # 編碼乾淨音訊（作為目標）
        with torch.no_grad():
            clean_features = model.encode(clean_audio.unsqueeze(1), input_lens=input_lens, use_quantizer=False)
        
        # 計算重建損失
        loss = F.mse_loss(noisy_features, clean_features)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
        })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """驗證"""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        noisy_audio = batch['noisy_audio'].to(device)
        clean_audio = batch['clean_audio'].to(device)
        
        # 計算音訊長度
        batch_size = noisy_audio.shape[0]
        input_lens = torch.tensor([noisy_audio.shape[1]] * batch_size, device=device)
        
        # 編碼
        noisy_features = model.encode(noisy_audio.unsqueeze(1), input_lens=input_lens, use_quantizer=False)
        clean_features = model.encode(clean_audio.unsqueeze(1), input_lens=input_lens, use_quantizer=False)
        
        # 計算損失
        loss = F.mse_loss(noisy_features, clean_features)
        total_loss += loss.item()
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiMo-Audio Encoder - Working Version")
    
    # 資料相關
    parser.add_argument('--data_dir', type=str, required=True,
                        help='訓練資料目錄')
    parser.add_argument('--tokenizer_path', type=str, 
                        default='./models/MiMo-Audio-Tokenizer',
                        help='Audio Tokenizer 模型路徑')
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/finetune_encoder',
                        help='輸出目錄')
    
    # LoRA 參數
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 80)
    logger.info("MiMo-Audio Encoder Fine-tuning - Working Version")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Device: {args.device}")
    
    # 載入 Audio Tokenizer
    logger.info("\nLoading Audio Tokenizer...")
    tokenizer_model = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer_model = tokenizer_model.to(args.device)
    
    # 凍結 Quantizer
    logger.info("Freezing quantizer...")
    for param in tokenizer_model.encoder.quantizer.parameters():
        param.requires_grad = False
    
    # 應用 LoRA
    logger.info(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_layers = apply_lora_to_encoder(
        tokenizer_model.encoder,
        rank=args.lora_rank,
        alpha=args.lora_alpha
    )
    
    # 準備資料
    logger.info("\nPreparing datasets...")
    train_dataset = AudioPairDataset(args.data_dir, split='train')
    val_dataset = AudioPairDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 優化器
    trainable_params = [p for p in tokenizer_model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # 訓練循環
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # 訓練
        train_loss = train_epoch(
            tokenizer_model.encoder,
            train_loader,
            optimizer,
            args.device,
            epoch
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 驗證
        val_loss = validate(tokenizer_model.encoder, val_loader, args.device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / 'best_model'
            save_path.mkdir(exist_ok=True)
            
            # 保存 LoRA 權重
            lora_state = {}
            for name, param in tokenizer_model.encoder.named_parameters():
                if 'lora' in name and param.requires_grad:
                    lora_state[name] = param.cpu()
            
            torch.save(lora_state, save_path / 'lora_weights.pt')
            logger.info(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # 定期保存 checkpoint
        if epoch % 2 == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}'
            save_path.mkdir(exist_ok=True)
            
            lora_state = {}
            for name, param in tokenizer_model.encoder.named_parameters():
                if 'lora' in name and param.requires_grad:
                    lora_state[name] = param.cpu()
            
            torch.save(lora_state, save_path / 'lora_weights.pt')
            logger.info(f"Saved checkpoint: epoch {epoch}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
