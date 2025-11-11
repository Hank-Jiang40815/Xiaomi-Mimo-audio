#!/usr/bin/env python3
"""
準備微調訓練資料

將音訊配對（noisy/clean）整理成訓練格式
"""

import json
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def prepare_data(source_dir, output_dir, train_ratio=0.8):
    """
    準備訓練資料
    
    Args:
        source_dir: 源資料目錄（包含 mix/ 和 spk1/）
        output_dir: 輸出目錄
        train_ratio: 訓練集比例
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # 創建輸出目錄結構
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'mix').mkdir(parents=True, exist_ok=True)
        (split_dir / 'spk1').mkdir(parents=True, exist_ok=True)
    
    # 掃描所有配對
    mix_dir = source_dir / 'mix'
    spk_dir = source_dir / 'spk1'
    
    if not mix_dir.exists() or not spk_dir.exists():
        raise ValueError(f"Source directory must contain mix/ and spk1/ subdirectories")
    
    # 找到所有配對
    pairs = []
    for mix_file in sorted(mix_dir.glob('*.wav')):
        base_name = mix_file.stem
        spk_file = spk_dir / f"{base_name}.wav"
        
        if spk_file.exists():
            pairs.append({
                'base_name': base_name,
                'mix_path': str(mix_file),
                'spk_path': str(spk_file)
            })
    
    print(f"Found {len(pairs)} audio pairs")
    
    # 分割訓練集和驗證集
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # 複製檔案並創建配對資訊
    def copy_split(pairs, split_dir, split_name):
        pair_info = []
        
        for pair in tqdm(pairs, desc=f"Preparing {split_name}"):
            base_name = pair['base_name']
            
            # 複製檔案
            mix_dst = split_dir / 'mix' / f"{base_name}.wav"
            spk_dst = split_dir / 'spk1' / f"{base_name}.wav"
            
            shutil.copy2(pair['mix_path'], mix_dst)
            shutil.copy2(pair['spk_path'], spk_dst)
            
            pair_info.append({
                'noisy': str(mix_dst),
                'clean': str(spk_dst)
            })
        
        # 保存配對資訊
        with open(split_dir / f"{split_name}_pairs.json", 'w') as f:
            json.dump(pair_info, f, indent=2)
        
        return pair_info
    
    train_info = copy_split(train_pairs, train_dir, 'train')
    val_info = copy_split(val_pairs, val_dir, 'val')
    
    # 保存統計資訊
    stats = {
        'total_pairs': len(pairs),
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'train_ratio': train_ratio
    }
    
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ Data preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源資料目錄（包含 mix/ 和 spk1/）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出目錄')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='訓練集比例 (default: 0.8)')
    
    args = parser.parse_args()
    
    prepare_data(args.source_dir, args.output_dir, args.train_ratio)


if __name__ == '__main__':
    main()
