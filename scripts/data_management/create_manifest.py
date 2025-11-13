#!/usr/bin/env python3
"""
建立音訊資料集的完整清單 (Manifest)

功能：
  - 掃描指定資料集目錄
  - 自動配對 noisy/clean 音檔
  - 提取檔案資訊（編號、語者、句子ID等）
  - 輸出 JSON 格式的 manifest 檔案

使用範例：
  # 建立 LDV 資料集清單
  python scripts/data_management/create_manifest.py \\
      --dataset ldv \\
      --noisy-dir examples/ldv/mix \\
      --clean-dir examples/ldv/spk \\
      --output data/manifests/ldv_manifest.json

  # 建立 Optical_denoised 資料集清單
  python scripts/data_management/create_manifest.py \\
      --dataset optical_denoised \\
      --noisy-dir examples/optical_denoised/mix \\
      --clean-dir examples/optical_denoised/spk1 \\
      --output data/manifests/optical_denoised_manifest.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import wave


def extract_info_from_filename(filename: str, dataset: str) -> Dict[str, Optional[str]]:
    """從檔名提取資訊"""
    info = {
        "speaker": None,
        "sentence_id": None,
        "noise_type": None,
    }
    
    if dataset == "ldv":
        # Noisy 格式: boy1_papercup_LDV_001.wav
        # Clean 格式: boy1_papercup_clean_001.wav
        match = re.match(r'(boy\d+)_([a-z]+)_(LDV|clean)_(\d+)\.wav', filename)
        if match:
            info["speaker"] = match.group(1)
            info["noise_type"] = match.group(2)
            info["sentence_id"] = match.group(4)  # 第4組是編號
    
    elif dataset.startswith("optical"):
        # Mix (noisy) 格式: boy1_WOLDV_001.wav, girl1_WOLDV_001.wav
        # Spk1 (clean) 格式: boy1_WOLDV_clean_001.wav, girl1_WOLDV_clean_001.wav
        match = re.match(r'((boy|girl)\d+)_([A-Za-z]+)_(clean_)?(\d+)\.wav', filename)
        if match:
            info["speaker"] = match.group(1)
            info["noise_type"] = match.group(3)
            info["sentence_id"] = match.group(5)  # 第5組是編號
    
    return info


def get_audio_duration(filepath: str) -> Optional[float]:
    """獲取音檔長度（秒）"""
    try:
        with wave.open(filepath, 'r') as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
            return round(duration, 2)
    except Exception as e:
        print(f"⚠️  無法讀取音檔長度: {filepath} - {e}")
        return None


def create_manifest(
    dataset_name: str,
    noisy_dir: str,
    clean_dir: str,
    output_path: str,
    compute_duration: bool = False
):
    """建立資料集 manifest"""
    
    print(f"📊 建立 {dataset_name} 資料集 Manifest...")
    print(f"   含噪音目錄: {noisy_dir}")
    print(f"   乾淨目錄: {clean_dir}")
    print()
    
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir) if clean_dir else None
    
    # 掃描含噪音音檔
    noisy_files = sorted([f for f in noisy_dir.glob("*.wav")])
    print(f"✓ 找到 {len(noisy_files)} 個含噪音音檔")
    
    # 掃描乾淨音檔
    clean_files_dict = {}
    if clean_dir and clean_dir.exists():
        clean_files = sorted([f for f in clean_dir.glob("*.wav")])
        print(f"✓ 找到 {len(clean_files)} 個乾淨音檔")
        
        # 建立 sentence_id -> clean_file 的對應
        for cf in clean_files:
            info = extract_info_from_filename(cf.name, dataset_name)
            if info["sentence_id"]:
                clean_files_dict[info["sentence_id"]] = cf
    else:
        print(f"⚠️  乾淨音檔目錄不存在: {clean_dir}")
    
    # 建立樣本列表
    samples = []
    matched_count = 0
    unmatched_count = 0
    
    for noisy_file in noisy_files:
        info = extract_info_from_filename(noisy_file.name, dataset_name)
        sentence_id = info["sentence_id"]
        
        # 尋找對應的乾淨音檔
        clean_file = clean_files_dict.get(sentence_id)
        if clean_file:
            matched_count += 1
        else:
            unmatched_count += 1
        
        sample = {
            "id": sentence_id,
            "sentence_id": sentence_id,
            "speaker": info["speaker"],
            "noise_type": info["noise_type"],
            "noisy_file": str(noisy_file),
            "clean_file": str(clean_file) if clean_file else None,
            "text": "",  # 預留給句子內容
            "duration": None,
            "noise_level": None  # 預留 (x60/x65/x70)
        }
        
        # 計算音檔長度 (可選)
        if compute_duration and noisy_file.exists():
            sample["duration"] = get_audio_duration(str(noisy_file))
        
        samples.append(sample)
    
    # 統計資訊
    print()
    print("📈 統計:")
    print(f"   • 總樣本數: {len(samples)}")
    print(f"   • 已配對: {matched_count} (有對應的乾淨音檔)")
    print(f"   • 未配對: {unmatched_count} (沒有對應的乾淨音檔)")
    
    # 建立 manifest
    manifest = {
        "dataset_name": dataset_name,
        "total_samples": len(samples),
        "matched_pairs": matched_count,
        "unmatched_samples": unmatched_count,
        "noisy_dir": str(noisy_dir),
        "clean_dir": str(clean_dir) if clean_dir else None,
        "samples": samples
    }
    
    # 儲存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"✅ Manifest 已儲存至: {output_path}")
    print(f"   檔案大小: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="建立音訊資料集的 Manifest 檔案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # LDV 資料集
  python create_manifest.py --dataset ldv \\
      --noisy-dir examples/ldv/mix \\
      --clean-dir examples/ldv/spk \\
      --output data/manifests/ldv_manifest.json
  
  # Optical_denoised 資料集 (計算音檔長度)
  python create_manifest.py --dataset optical_denoised \\
      --noisy-dir examples/optical_denoised/mix \\
      --clean-dir examples/optical_denoised/spk1 \\
      --output data/manifests/optical_denoised_manifest.json \\
      --compute-duration
        """
    )
    
    parser.add_argument("--dataset", required=True,
                        help="資料集名稱 (ldv, optical, optical_denoised, 等)")
    parser.add_argument("--noisy-dir", required=True,
                        help="含噪音音檔目錄")
    parser.add_argument("--clean-dir",
                        help="乾淨音檔目錄 (可選)")
    parser.add_argument("--output", required=True,
                        help="輸出 manifest JSON 檔案路徑")
    parser.add_argument("--compute-duration", action="store_true",
                        help="計算音檔長度 (會花費較多時間)")
    
    args = parser.parse_args()
    
    create_manifest(
        dataset_name=args.dataset,
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        output_path=args.output,
        compute_duration=args.compute_duration
    )


if __name__ == "__main__":
    main()
