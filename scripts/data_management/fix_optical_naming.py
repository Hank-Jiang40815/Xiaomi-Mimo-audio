#!/usr/bin/env python3
"""
修正 optical 資料集的檔案命名

問題: 部分檔案命名不符合標準格式
  - 標準格式: boy1_WOLDVlean_001.wav (有底線)
  - 非標準格式: girl3WOLDV121.wav (沒底線)

解決: 將非標準格式重命名為標準格式
  girl3WOLDV121.wav -> girl3_WOLDV_121.wav
  
使用方式:
  # 預覽重命名 (不實際執行)
  python scripts/data_management/fix_optical_naming.py --dry-run
  
  # 實際執行重命名
  python scripts/data_management/fix_optical_naming.py
"""

import argparse
import re
from pathlib import Path


def fix_filename(filename: str) -> str:
    """
    修正檔案名稱格式
    
    輸入: girl3WOLDV121.wav 或 boy3_WOLDV001.wav
    輸出: girl3_WOLDV_121.wav 或 boy3_WOLDV_001.wav
    """
    # 檢查是否已經是標準格式 (兩個底線)
    if re.match(r'(boy|girl)\d+_[A-Za-z]+_\d+\.wav', filename):
        return None  # 已經是標準格式，不需要修改
    
    # 格式 1: 沒有底線 - girl3WOLDV121.wav
    match = re.match(r'((boy|girl)\d+)([A-Za-z]+)(\d+)\.wav', filename)
    if match:
        speaker = match.group(1)  # girl3
        noise_type = match.group(3)  # WOLDV
        number = match.group(4)  # 121
        
        # 格式化為標準格式
        new_filename = f"{speaker}_{noise_type}_{number}.wav"
        return new_filename
    
    # 格式 2: 只有一個底線 - boy3_WOLDV001.wav
    match = re.match(r'((boy|girl)\d+)_([A-Za-z]+)(\d+)\.wav', filename)
    if match:
        speaker = match.group(1)  # boy3
        noise_type = match.group(3)  # WOLDV
        number = match.group(4)  # 001
        
        # 格式化為標準格式
        new_filename = f"{speaker}_{noise_type}_{number}.wav"
        return new_filename
    
    return None


def rename_files(directory: Path, dry_run: bool = True):
    """重命名目錄中的檔案"""
    
    files = sorted(directory.glob("*.wav"))
    renamed_count = 0
    skipped_count = 0
    
    print(f"📁 處理目錄: {directory}")
    print(f"   總檔案數: {len(files)}")
    print()
    
    if dry_run:
        print("⚠️  預覽模式 (不會實際重命名)")
    else:
        print("✅ 執行模式 (將實際重命名檔案)")
    print()
    
    for file_path in files:
        old_name = file_path.name
        new_name = fix_filename(old_name)
        
        if new_name:
            new_path = file_path.parent / new_name
            
            if dry_run:
                print(f"  {old_name}")
                print(f"  → {new_name}")
                print()
            else:
                file_path.rename(new_path)
                print(f"✓ {old_name} → {new_name}")
            
            renamed_count += 1
        else:
            skipped_count += 1
    
    print()
    print("=" * 70)
    print(f"總計:")
    print(f"  • 需要重命名: {renamed_count} 個檔案")
    print(f"  • 無需修改:   {skipped_count} 個檔案")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="修正 optical 資料集的檔案命名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 預覽重命名 (推薦先執行):
   python fix_optical_naming.py --dry-run

2. 實際執行重命名:
   python fix_optical_naming.py

3. 只處理 mix 目錄:
   python fix_optical_naming.py --dir examples/optical/mix

4. 只處理 spk1 目錄:
   python fix_optical_naming.py --dir examples/optical/spk1
        """
    )
    
    parser.add_argument("--dir", 
                        default="examples/optical/mix",
                        help="要處理的目錄 (預設: examples/optical/mix)")
    parser.add_argument("--dry-run", 
                        action="store_true",
                        help="預覽模式，不實際重命名")
    
    args = parser.parse_args()
    
    directory = Path(args.dir)
    
    if not directory.exists():
        print(f"❌ 錯誤: 目錄不存在: {directory}")
        return 1
    
    if not directory.is_dir():
        print(f"❌ 錯誤: 不是目錄: {directory}")
        return 1
    
    # 執行重命名
    rename_files(directory, dry_run=args.dry_run)
    
    if args.dry_run:
        print()
        print("💡 提示: 使用 --dry-run 可以不加參數直接執行重命名")
    
    return 0


if __name__ == "__main__":
    exit(main())
