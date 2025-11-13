#!/usr/bin/env python3
"""
快速測試：驗證資料載入是否正確
"""
import sys
import json
from pathlib import Path

def test_data_loading():
    """測試資料載入"""
    print("🧪 測試資料載入...")
    print()
    
    # 測試 split 檔案
    split_dir = Path("data/splits/finetune_optical")
    
    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f"{split_name}.json"
        
        if not split_file.exists():
            print(f"❌ {split_file} 不存在!")
            return False
        
        with open(split_file) as f:
            data = json.load(f)
        
        if 'samples' not in data:
            print(f"❌ {split_file} 格式錯誤，缺少 'samples' 鍵")
            return False
        
        samples = data['samples']
        print(f"✅ {split_name.upper()}: {len(samples)} samples")
        
        # 檢查第一個樣本
        if len(samples) > 0:
            sample = samples[0]
            required_keys = ['noisy_file', 'clean_file']
            missing = [k for k in required_keys if k not in sample]
            if missing:
                print(f"   ⚠️  缺少鍵: {missing}")
                return False
            
            # 檢查檔案是否存在
            noisy_path = Path(sample['noisy_file'])
            clean_path = Path(sample['clean_file'])
            
            if not noisy_path.exists():
                print(f"   ⚠️  噪聲檔案不存在: {noisy_path}")
                return False
            
            if not clean_path.exists():
                print(f"   ⚠️  乾淨檔案不存在: {clean_path}")
                return False
            
            print(f"   • Noisy: {noisy_path.name}")
            print(f"   • Clean: {clean_path.name}")
            if 'text' in sample and sample['text']:
                text_preview = sample['text'][:30] + '...' if len(sample['text']) > 30 else sample['text']
                print(f"   • Text: {text_preview}")
        print()
    
    print("✅ 所有資料檢查通過!")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
