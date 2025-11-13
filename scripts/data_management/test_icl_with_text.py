#!/usr/bin/env python3
"""
測試包含文字標註的 ICL 功能

使用 optical manifest 測試 In-Context Learning 的完整流程：
1. 從 manifest 載入樣本
2. 建立 prompt_examples (包含文字標註)
3. 執行 ICL 推論
"""

import json
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mimo_audio.mimo_audio import MimoAudio


def load_samples_from_manifest(manifest_path: str, max_samples: int = 5):
    """從 manifest 載入樣本"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    samples = manifest["samples"]
    
    # 過濾出有 clean 檔案和文字標註的樣本
    valid_samples = [
        s for s in samples 
        if s["clean_file"] and s["text"] and len(s["text"].strip()) > 0
    ]
    
    print(f"📊 Manifest 統計:")
    print(f"   總樣本數: {len(samples)}")
    print(f"   有效樣本 (有 clean + text): {len(valid_samples)}")
    print()
    
    if len(valid_samples) < max_samples:
        print(f"⚠️  有效樣本數不足 {max_samples}，使用所有 {len(valid_samples)} 個樣本")
        return valid_samples
    
    return valid_samples[:max_samples]


def main():
    # 路徑設定
    manifest_path = "data/manifests/optical_manifest.json"
    model_path = "models/MiMo-Audio-7B-Base"
    tokenizer_path = "models/MiMo-Audio-Tokenizer"
    output_audio_path = "examples/optical_icl_test_result.wav"
    
    print("="*60)
    print("測試包含文字標註的 ICL 功能")
    print("="*60)
    print()
    
    # 1. 載入樣本
    print("📂 載入 manifest...")
    samples = load_samples_from_manifest(manifest_path, max_samples=5)
    
    if len(samples) < 2:
        print("❌ 錯誤: 至少需要 2 個有效樣本 (1 個用於測試 + 1 個用於 few-shot)")
        return
    
    # 2. 準備 few-shot examples (使用前 2 個)
    prompt_examples = []
    for i, sample in enumerate(samples[:2], 1):
        prompt_examples.append({
            "input_audio": sample["noisy_file"],
            "output_audio": sample["clean_file"],
            "output_transcription": sample["text"],
        })
        print(f"📝 Few-shot 範例 {i}:")
        print(f"   輸入: {Path(sample['noisy_file']).name}")
        print(f"   輸出: {Path(sample['clean_file']).name}")
        print(f"   文字: {sample['text'][:50]}...")
        print()
    
    # 3. 準備測試輸入 (使用第 3 個)
    test_sample = samples[2]
    input_audio = test_sample["noisy_file"]
    
    print(f"🎯 測試樣本:")
    print(f"   輸入: {Path(input_audio).name}")
    print(f"   參考 clean: {Path(test_sample['clean_file']).name}")
    print(f"   參考文字: {test_sample['text']}")
    print()
    
    # 4. 載入模型
    print("📥 載入 MiMo-Audio Base 模型...")
    model = MimoAudio(model_path, tokenizer_path)
    print("✅ 模型載入完成")
    print()
    
    # 5. 執行 ICL
    instruction = "Enhance the audio quality and remove noise from the input speech."
    
    print("="*60)
    print("開始 ICL 推論...")
    print("="*60)
    print()
    
    try:
        text_output = model.in_context_learning_s2s(
            instruction,
            prompt_examples,
            input_audio,
            max_new_tokens=8192,
            output_audio_path=output_audio_path
        )
        
        print()
        print("="*60)
        print("✅ ICL 推論完成！")
        print("="*60)
        print(f"📝 模型輸出文字: {text_output}")
        print(f"🎵 輸出音檔: {output_audio_path}")
        print()
        print("📊 比較:")
        print(f"   參考文字: {test_sample['text']}")
        print(f"   模型輸出: {text_output}")
        print()
        
    except Exception as e:
        print()
        print("="*60)
        print("❌ ICL 推論失敗！")
        print("="*60)
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
