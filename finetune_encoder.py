#!/usr/bin/env python3
"""
MiMo-Audio Encoder Fine-tuning Script
針對嚴重噪聲場景微調 Encoder

使用 LoRA (Low-Rank Adaptation) 進行高效微調
重點：Audio Tokenizer 的 Encoder 部分
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datetime import datetime

# 設置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPairDataset(Dataset):
    """
    音訊配對資料集
    - noisy_audio: 噪聲音訊路徑
    - clean_audio: 乾淨音訊路徑
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 優先載入 split_selector 生成的格式
        split_file = self.data_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            # 檢查是否是 split_selector 格式（有 'samples' 鍵）
            if 'samples' in split_data:
                self.pairs = self._load_from_manifest_split(split_data)
                logger.info(f"Loaded {len(self.pairs)} pairs from manifest split")
            else:
                self.pairs = split_data
                logger.info(f"Loaded {len(self.pairs)} pairs from legacy format")
        else:
            # 舊格式：{split}_pairs.json
            pair_file = self.data_dir / f"{split}_pairs.json"
            if pair_file.exists():
                with open(pair_file, 'r') as f:
                    self.pairs = json.load(f)
                logger.info(f"Loaded {len(self.pairs)} pairs from {pair_file}")
            else:
                # 自動掃描配對
                self.pairs = self._auto_scan_pairs()
                logger.info(f"Auto-scanned {len(self.pairs)} pairs")
    
    def _load_from_manifest_split(self, split_data):
        """從 split_selector 生成的 manifest 格式載入"""
        pairs = []
        for sample in split_data['samples']:
            pairs.append({
                'noisy': sample['noisy_file'],
                'clean': sample['clean_file'],
                'text': sample.get('text', ''),
                'speaker': sample.get('speaker', ''),
                'id': sample.get('id', '')
            })
        return pairs
    
    def _auto_scan_pairs(self):
        """自動掃描 mix/ 和 spk/ 目錄找配對"""
        pairs = []
        mix_dir = self.data_dir / "mix"
        spk_dir = self.data_dir / "spk1"
        
        if mix_dir.exists() and spk_dir.exists():
            for mix_file in sorted(mix_dir.glob("*.wav")):
                # 找對應的乾淨音訊
                base_name = mix_file.stem
                spk_file = spk_dir / f"{base_name}.wav"
                
                if spk_file.exists():
                    pairs.append({
                        "noisy": str(mix_file),
                        "clean": str(spk_file)
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'noisy_path': pair['noisy'],
            'clean_path': pair['clean']
        }


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    
    對原始權重矩陣 W 添加低秩分解: W' = W + BA
    其中 B: (d_out, r), A: (r, d_in), r << min(d_in, d_out)
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # 凍結原始層
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # 獲取原始層的輸入輸出維度
        if hasattr(original_layer, 'in_features'):
            d_in = original_layer.in_features
            d_out = original_layer.out_features
        elif hasattr(original_layer, 'in_channels'):
            d_in = original_layer.in_channels
            d_out = original_layer.out_channels
        else:
            raise ValueError("Unsupported layer type for LoRA")
        
        # LoRA 矩陣
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始輸出
        result = self.original_layer(x)
        
        # LoRA 調整
        if isinstance(self.original_layer, nn.Linear):
            lora_adjustment = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        else:
            # 對於 Conv 層的處理
            lora_adjustment = torch.nn.functional.conv1d(
                x, 
                (self.lora_B @ self.lora_A).unsqueeze(-1),
                bias=None
            ) * self.scaling
        
        return result + lora_adjustment


def apply_lora_to_encoder(tokenizer_model, rank=8, alpha=16):
    """
    對 Audio Tokenizer 的 Encoder 應用 LoRA
    
    Args:
        tokenizer_model: MiMo Audio Tokenizer model
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
    """
    logger.info(f"Applying LoRA to encoder (rank={rank}, alpha={alpha})")
    
    # 找到 encoder 的所有線性層和卷積層
    lora_layers = []
    
    if hasattr(tokenizer_model, 'encoder'):
        encoder = tokenizer_model.encoder
        
        for name, module in encoder.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # 替換為 LoRA 層
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(encoder.named_modules())[parent_name]
                else:
                    parent = encoder
                
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_layers.append(lora_layer)
                
                logger.info(f"Applied LoRA to: {name}")
    
    logger.info(f"Total LoRA layers applied: {len(lora_layers)}")
    return lora_layers


def compute_reconstruction_loss(noisy_tokens, clean_tokens):
    """
    計算重建損失
    
    Args:
        noisy_tokens: 噪聲音訊的 token representation
        clean_tokens: 乾淨音訊的 token representation
    """
    # L1 loss (更適合音訊重建)
    l1_loss = nn.functional.l1_loss(noisy_tokens, clean_tokens)
    
    # MSE loss
    mse_loss = nn.functional.mse_loss(noisy_tokens, clean_tokens)
    
    # 組合損失
    total_loss = 0.7 * l1_loss + 0.3 * mse_loss
    
    return total_loss, {'l1': l1_loss.item(), 'mse': mse_loss.item()}


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        noisy_paths = batch['noisy_path']
        clean_paths = batch['clean_path']
        
        # TODO: 載入和預處理音訊
        # 這裡需要根據實際的音訊載入方式來實現
        # noisy_audio = load_audio(noisy_paths)
        # clean_audio = load_audio(clean_paths)
        
        # 前向傳播（通過 encoder）
        # noisy_tokens = model.encode(noisy_audio)
        # clean_tokens = model.encode(clean_audio)
        
        # 計算損失
        # loss, loss_dict = compute_reconstruction_loss(noisy_tokens, clean_tokens)
        
        # 反向傳播
        optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        # scheduler.step()
        
        # total_loss += loss.item()
        
        # 更新進度條
        progress_bar.set_postfix({
            'loss': f"{total_loss/(batch_idx+1):.4f}"
        })
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """驗證"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            noisy_paths = batch['noisy_path']
            clean_paths = batch['clean_path']
            
            # TODO: 同上，實現音訊載入和驗證邏輯
            pass
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiMo-Audio Encoder")
    
    # 資料相關
    parser.add_argument('--data_dir', type=str, required=True,
                        help='訓練資料目錄 (包含 mix/ 和 spk1/ 子目錄)')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='驗證資料目錄')
    
    # 模型相關
    parser.add_argument('--tokenizer_path', type=str, 
                        default='./models/MiMo-Audio-Tokenizer',
                        help='Audio Tokenizer 模型路徑')
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/finetune_encoder',
                        help='輸出目錄')
    
    # LoRA 參數
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='訓練輪數')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='每多少步保存一次')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("=" * 80)
    logger.info("MiMo-Audio Encoder Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    
    # TODO: 載入 Audio Tokenizer
    # from src.mimo_audio_tokenizer import AudioTokenizer
    # tokenizer_model = AudioTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer_model = tokenizer_model.to(args.device)
    
    # 應用 LoRA
    # lora_layers = apply_lora_to_encoder(
    #     tokenizer_model, 
    #     rank=args.lora_rank, 
    #     alpha=args.lora_alpha
    # )
    
    # 準備資料
    train_dataset = AudioPairDataset(args.data_dir, split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    if args.val_data_dir:
        val_dataset = AudioPairDataset(args.val_data_dir, split='val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        val_loader = None
    
    # 優化器和調度器
    # trainable_params = [p for p in tokenizer_model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay
    # )
    
    # total_steps = len(train_loader) * args.epochs
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=total_steps
    # )
    
    # 訓練循環
    # best_val_loss = float('inf')
    
    # for epoch in range(1, args.epochs + 1):
    #     logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
    #     # 訓練
    #     train_loss = train_epoch(
    #         tokenizer_model, train_loader, optimizer, scheduler,
    #         args.device, epoch
    #     )
    #     logger.info(f"Train Loss: {train_loss:.4f}")
        
    #     # 驗證
    #     if val_loader:
    #         val_loss = validate(tokenizer_model, val_loader, args.device)
    #         logger.info(f"Val Loss: {val_loss:.4f}")
            
    #         # 保存最佳模型
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             save_path = output_dir / 'best_model'
    #             tokenizer_model.save_pretrained(save_path)
    #             logger.info(f"Saved best model to {save_path}")
        
    #     # 定期保存
    #     if epoch % 2 == 0:
    #         save_path = output_dir / f'checkpoint_epoch_{epoch}'
    #         tokenizer_model.save_pretrained(save_path)
    #         logger.info(f"Saved checkpoint to {save_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    logger.info("\n⚠️  注意: 這是一個框架腳本，需要根據實際的模型 API 來完成實現")
    logger.info("主要需要實現的部分:")
    logger.info("  1. 載入 Audio Tokenizer 模型")
    logger.info("  2. 音訊載入和預處理")
    logger.info("  3. Encoder 的前向傳播")
    logger.info("  4. 完整的訓練和驗證循環")


if __name__ == '__main__':
    main()
