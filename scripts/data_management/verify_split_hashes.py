#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查資料分割的 hash 記錄是否與實際檔案一致。

使用方式:
    python scripts/data_management/verify_split_hashes.py \\
        --split-dir data/splits/finetune_optical
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """計算檔案的 SHA256。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def combined_hash(entries: List[Dict[str, Dict[str, str]]]) -> str:
    """依固定順序計算 combined hash。"""
    h = hashlib.sha256()
    for item in sorted(
        entries,
        key=lambda e: (e["noisy"]["path"], e["clean"]["path"]),
    ):
        for side in ("noisy", "clean"):
            h.update(item[side]["path"].encode("utf-8"))
            h.update(item[side]["sha256"].encode("utf-8"))
    return h.hexdigest()


def load_split_samples(split_path: Path) -> List[Dict]:
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    return split_data.get("samples", [])


def verify_split(split_name: str, split_dir: Path, recorded: Dict) -> Tuple[bool, str]:
    split_path = split_dir / f"{split_name}.json"
    if not split_path.exists():
        return False, f"[{split_name}] split file not found: {split_path}"

    samples = load_split_samples(split_path)
    expected_total = recorded.get("total", 0)
    if len(samples) != expected_total:
        return False, f"[{split_name}] total mismatch: {len(samples)} != {expected_total}"

    recorded_entries = recorded.get("entries", [])
    lookup = {
        (entry["noisy"]["path"], entry["clean"]["path"]): entry
        for entry in recorded_entries
    }

    recomputed_entries: List[Dict[str, Dict[str, str]]] = []
    for sample in samples:
        key = (sample["noisy_file"], sample["clean_file"])
        stored_entry = lookup.get(key)
        if stored_entry is None:
            return False, f"[{split_name}] entry not found in hash record: {key}"

        noisy_path = Path(sample["noisy_file"])
        clean_path = Path(sample["clean_file"])
        if not noisy_path.exists():
            return False, f"[{split_name}] noisy file missing: {noisy_path}"
        if not clean_path.exists():
            return False, f"[{split_name}] clean file missing: {clean_path}"

        noisy_hash = sha256(noisy_path)
        clean_hash = sha256(clean_path)
        if noisy_hash != stored_entry["noisy"]["sha256"]:
            return (
                False,
                f"[{split_name}] noisy hash mismatch for {noisy_path}: "
                f"{noisy_hash} != {stored_entry['noisy']['sha256']}",
            )
        if clean_hash != stored_entry["clean"]["sha256"]:
            return (
                False,
                f"[{split_name}] clean hash mismatch for {clean_path}: "
                f"{clean_hash} != {stored_entry['clean']['sha256']}",
            )
        recomputed_entries.append(
            {
                "noisy": {"path": sample["noisy_file"], "sha256": noisy_hash},
                "clean": {"path": sample["clean_file"], "sha256": clean_hash},
            }
        )

    recomputed_hash = combined_hash(recomputed_entries)
    recorded_hash = recorded.get("combined_hash")
    if recomputed_hash != recorded_hash:
        return (
            False,
            f"[{split_name}] combined hash mismatch: {recomputed_hash} != {recorded_hash}",
        )

    return True, f"[{split_name}] OK ({len(samples)} samples)"


def main():
    parser = argparse.ArgumentParser(description="驗證 split_hashes.json 記錄")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/finetune_optical"),
        help="train/val/test JSON 所在目錄",
    )
    parser.add_argument(
        "--hash-file",
        type=Path,
        help="split_hashes.json 路徑（預設為 <split-dir>/split_hashes.json）",
    )
    args = parser.parse_args()

    hash_file = args.hash_file or (args.split_dir / "split_hashes.json")
    if not hash_file.exists():
        print(f"❌ Hash file not found: {hash_file}", file=sys.stderr)
        sys.exit(1)

    with hash_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data.get("splits", {})
    if not splits:
        print("❌ No split entries found in hash file", file=sys.stderr)
        sys.exit(1)

    all_ok = True
    for split_name, recorded in splits.items():
        ok, message = verify_split(split_name, args.split_dir, recorded)
        status = "✅" if ok else "❌"
        print(f"{status} {message}")
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
