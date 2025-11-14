#!/usr/bin/env python3
"""
快速測試微調腳本 - 使用少量數據驗證流程
"""

import sys
import json
from pathlib import Path

# 創建小型測試 splits
def create_test_splits(original_split_dir, test_split_dir, num_train=10, num_val=5):
    """從原始 splits 創建小型測試 splits"""
    
    original_split_dir = Path(original_split_dir)
    test_split_dir = Path(test_split_dir)
    test_split_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入原始資料
    with open(original_split_dir / 'train.json', 'r') as f:
        train_data = json.load(f)
    
    with open(original_split_dir / 'val.json', 'r') as f:
        val_data = json.load(f)
    
    # 取少量樣本
    test_train = {
        'dataset': train_data['dataset'],
        'split': 'train_test',
        'num_samples': num_train,
        'samples': train_data['samples'][:num_train]
    }
    
    test_val = {
        'dataset': val_data['dataset'],
        'split': 'val_test',
        'num_samples': num_val,
        'samples': val_data['samples'][:num_val]
    }
    
    # 保存測試 splits
    with open(test_split_dir / 'train.json', 'w') as f:
        json.dump(test_train, f, indent=2, ensure_ascii=False)
    
    with open(test_split_dir / 'val.json', 'w') as f:
        json.dump(test_val, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created test splits:")
    print(f"   Train: {num_train} samples -> {test_split_dir / 'train.json'}")
    print(f"   Val: {num_val} samples -> {test_split_dir / 'val.json'}")


if __name__ == '__main__':
    # 創建測試 splits
    create_test_splits(
        original_split_dir='data/splits/finetune_optical',
        test_split_dir='data/splits/finetune_optical_test',
        num_train=10,
        num_val=5
    )
    
    print("\n🧪 Test splits created!")
    print("\n📝 Run quick test with:")
    print("   docker run --gpus all --rm -v \"$(pwd)\":/workspace -w /workspace mimo-audio:latest \\")
    print("     python finetune_encoder.py \\")
    print("       --train-split data/splits/finetune_optical_test/train.json \\")
    print("       --val-split data/splits/finetune_optical_test/val.json \\")
    print("       --batch-size 2 \\")
    print("       --gradient-accumulation-steps 2 \\")
    print("       --epochs 2 \\")
    print("       --lora-rank 8 \\")
    print("       --output-dir ./outputs/finetune_test")
