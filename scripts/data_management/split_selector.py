#!/usr/bin/env python3
"""
資料分割選擇器 - 靈活選擇 ICL examples 或微調資料集

功能：
  - 多種選擇模式：sequential, random, custom range
  - 支援自訂編號範圍 (如: "1-40,50-60,100")
  - ICL 實驗用 (10/40/80/160-shot 等)
  - 微調資料集分割 (train/val/test)
  - 輸出 JSON 格式的 split 檔案

使用範例：
  # ICL - 選擇前 40 個
  python split_selector.py --manifest data/manifests/ldv_manifest.json \\
      --mode icl --shots 40 --method sequential \\
      --output data/splits/icl/40shot.json

  # ICL - 自訂範圍
  python split_selector.py --manifest data/manifests/ldv_manifest.json \\
      --mode icl --custom "1-20,50-60,100-109" \\
      --output data/splits/icl/custom_30shot.json

  # ICL - 隨機選擇
  python split_selector.py --manifest data/manifests/ldv_manifest.json \\
      --mode icl --shots 40 --method random --seed 42 \\
      --output data/splits/icl/40shot_random.json

  # 微調 - 8:1:1 分割
  python split_selector.py --manifest data/manifests/ldv_manifest.json \\
      --mode finetune --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \\
      --output-dir data/splits/finetune/
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def load_manifest(manifest_path: str) -> Dict:
    """載入 manifest 檔案"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_range_string(range_str: str) -> List[int]:
    """
    解析範圍字串，回傳數字列表
    
    範例:
      "1-10" -> [1, 2, 3, ..., 10]
      "1-5,10,15-20" -> [1,2,3,4,5,10,15,16,17,18,19,20]
    """
    result = []
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 範圍 (例如: "1-10")
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            # 單一數字
            result.append(int(part))
    
    return sorted(list(set(result)))  # 去重複並排序


def select_samples_icl(
    samples: List[Dict],
    shots: Optional[int] = None,
    method: str = "sequential",
    custom_range: Optional[str] = None,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    選擇 ICL examples
    
    Args:
        samples: 所有樣本
        shots: 要選擇幾個 shots
        method: 選擇方法 (sequential, random)
        custom_range: 自訂範圍字串 (例如: "1-40,50-60")
        seed: 隨機種子
    """
    
    if custom_range:
        # 自訂範圍模式
        indices = parse_range_string(custom_range)
        print(f"📌 自訂範圍: {len(indices)} 個樣本")
        print(f"   範圍: {custom_range}")
        
        selected = []
        for idx in indices:
            # 找到對應的樣本 (sentence_id 匹配)
            for sample in samples:
                if sample["sentence_id"] == f"{idx:03d}":
                    selected.append(sample)
                    break
        
        return selected
    
    elif method == "sequential":
        # 順序選擇前 N 個
        print(f"📌 順序選擇: 前 {shots} 個樣本")
        return samples[:shots]
    
    elif method == "random":
        # 隨機選擇
        if seed is not None:
            random.seed(seed)
        print(f"📌 隨機選擇: {shots} 個樣本 (seed={seed})")
        return random.sample(samples, shots)
    
    else:
        raise ValueError(f"未知的選擇方法: {method}")


def split_for_finetune(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分割資料集用於微調
    
    Returns:
        (train_samples, val_samples, test_samples)
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "比例總和必須為 1.0"
    
    total = len(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    print(f"📌 微調資料分割:")
    print(f"   • 總樣本: {total}")
    print(f"   • 訓練集: {train_size} ({train_ratio*100:.0f}%)")
    print(f"   • 驗證集: {val_size} ({val_ratio*100:.0f}%)")
    print(f"   • 測試集: {test_size} ({test_ratio*100:.0f}%)")
    
    # 打亂樣本
    if seed is not None:
        random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    train_samples = shuffled[:train_size]
    val_samples = shuffled[train_size:train_size + val_size]
    test_samples = shuffled[train_size + val_size:]
    
    return train_samples, val_samples, test_samples


def save_split(split_data: Dict, output_path: str):
    """儲存 split 檔案"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Split 已儲存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="資料分割選擇器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. ICL - 順序選擇前 40 個:
   python split_selector.py --manifest data/manifests/ldv_manifest.json \\
       --mode icl --shots 40 --method sequential \\
       --output data/splits/icl/40shot.json

2. ICL - 自訂範圍 (1-20, 50-60, 100-109):
   python split_selector.py --manifest data/manifests/ldv_manifest.json \\
       --mode icl --custom "1-20,50-60,100-109" \\
       --output data/splits/icl/custom_30shot.json

3. ICL - 隨機選擇 80 個:
   python split_selector.py --manifest data/manifests/ldv_manifest.json \\
       --mode icl --shots 80 --method random --seed 42 \\
       --output data/splits/icl/80shot_random.json

4. 微調 - 8:1:1 分割:
   python split_selector.py --manifest data/manifests/ldv_manifest.json \\
       --mode finetune --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \\
       --output-dir data/splits/finetune/ --seed 42
        """
    )
    
    parser.add_argument("--manifest", required=True,
                        help="Manifest 檔案路徑")
    parser.add_argument("--mode", required=True, choices=["icl", "finetune"],
                        help="模式: icl (ICL 實驗) 或 finetune (微調)")
    
    # ICL 模式參數
    parser.add_argument("--shots", type=int,
                        help="ICL shot 數量")
    parser.add_argument("--method", choices=["sequential", "random"],
                        default="sequential",
                        help="選擇方法 (預設: sequential)")
    parser.add_argument("--custom",
                        help='自訂範圍字串 (例如: "1-40,50-60,100")')
    
    # 微調模式參數
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="訓練集比例 (預設: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="驗證集比例 (預設: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="測試集比例 (預設: 0.1)")
    
    # 通用參數
    parser.add_argument("--seed", type=int,
                        help="隨機種子")
    parser.add_argument("--output",
                        help="輸出檔案路徑 (ICL 模式)")
    parser.add_argument("--output-dir",
                        help="輸出目錄 (微調模式，會產生 train.json, val.json, test.json)")
    
    args = parser.parse_args()
    
    # 載入 manifest
    print(f"📂 載入 Manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    samples = manifest["samples"]
    print(f"   總樣本數: {len(samples)}")
    print()
    
    if args.mode == "icl":
        # ICL 模式
        if not args.output:
            parser.error("ICL 模式需要指定 --output")
        
        if args.custom:
            selected_samples = select_samples_icl(
                samples, custom_range=args.custom
            )
        elif args.shots:
            selected_samples = select_samples_icl(
                samples, shots=args.shots, method=args.method, seed=args.seed
            )
        else:
            parser.error("ICL 模式需要指定 --shots 或 --custom")
        
        # 建立 split 資料
        split_data = {
            "dataset": manifest["dataset_name"],
            "mode": "icl",
            "total_samples": len(selected_samples),
            "method": "custom" if args.custom else args.method,
            "range": args.custom if args.custom else f"1-{args.shots}",
            "seed": args.seed,
            "samples": selected_samples
        }
        
        save_split(split_data, args.output)
    
    elif args.mode == "finetune":
        # 微調模式
        if not args.output_dir:
            parser.error("微調模式需要指定 --output-dir")
        
        train_samples, val_samples, test_samples = split_for_finetune(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        output_dir = Path(args.output_dir)
        
        # 儲存訓練集
        save_split({
            "dataset": manifest["dataset_name"],
            "mode": "finetune",
            "split": "train",
            "total_samples": len(train_samples),
            "samples": train_samples
        }, output_dir / "train.json")
        
        # 儲存驗證集
        save_split({
            "dataset": manifest["dataset_name"],
            "mode": "finetune",
            "split": "val",
            "total_samples": len(val_samples),
            "samples": val_samples
        }, output_dir / "val.json")
        
        # 儲存測試集
        save_split({
            "dataset": manifest["dataset_name"],
            "mode": "finetune",
            "split": "test",
            "total_samples": len(test_samples),
            "samples": test_samples
        }, output_dir / "test.json")
    
    print()
    print("✅ 完成！")


if __name__ == "__main__":
    main()
