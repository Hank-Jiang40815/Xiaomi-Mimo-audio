#!/usr/bin/env python3
"""
Optical Dataset ICL Testing
使用 In-Context Learning 測試 optical 資料集
"""

import sys
import json
import argparse
from pathlib import Path
import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mimo_audio.mimo_audio import MimoAudio


def load_test_split(split_file):
    """載入測試集"""
    with open(split_file) as f:
        data = json.load(f)
    
    if 'samples' not in data:
        raise ValueError("Invalid split format")
    
    return data['samples']


def select_prompt_examples(train_split, num_shots, seed=42):
    """從訓練集選擇 prompt examples"""
    with open(train_split) as f:
        train_data = json.load(f)
    
    samples = train_data['samples']
    random.seed(seed)
    selected = random.sample(samples, min(num_shots, len(samples)))
    
    prompt_examples = []
    for s in selected:
        prompt_examples.append({
            "input_audio": s['noisy_file'],
            "output_audio": s['clean_file'],
            "output_transcription": s.get('text', '')  # MiMo-Audio 需要這個欄位名稱
        })
    
    return prompt_examples


def test_icl(args):
    """執行 ICL 測試"""
    print("=" * 80)
    print(f"Optical Dataset ICL Testing - {args.shots}-shot")
    print("=" * 80)
    
    # 載入模型
    print("\n載入模型...")
    model = MimoAudio(
        args.model_path,
        args.tokenizer_path,
        device=args.device
    )
    
    # 載入測試集
    print(f"載入測試集: {args.test_split}")
    test_samples = load_test_split(args.test_split)
    print(f"測試樣本數: {len(test_samples)}")
    
    # 選擇 prompt examples
    train_split = str(Path(args.test_split).parent / "train.json")
    print(f"\n從訓練集選擇 {args.shots} 個 prompt examples")
    prompt_examples = select_prompt_examples(train_split, args.shots)
    
    print("\nPrompt Examples:")
    for i, ex in enumerate(prompt_examples):
        print(f"  {i+1}. {Path(ex['input_audio']).name}")
        if ex.get('output_transcription'):
            text_preview = ex['output_transcription'][:30] + '...' if len(ex['output_transcription']) > 30 else ex['output_transcription']
            print(f"     Text: {text_preview}")
    
    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 限制測試樣本數
    test_samples_to_process = test_samples[:args.num_samples]
    print(f"\n處理 {len(test_samples_to_process)} 個測試樣本")
    
    # 測試每個樣本
    results = []
    
    for idx, sample in enumerate(tqdm(test_samples_to_process, desc="Testing")):
        try:
            # 準備輸入
            input_audio = sample['noisy_file']
            expected_output = sample['clean_file']
            text = sample.get('text', '')
            
            # ICL 推論
            instruction = f"請增強這段語音，內容是：{text}" if text else "請增強這段語音的品質"
            
            output_audio_path = output_dir / f"{sample['id']}_{args.shots}shot_enhanced.wav"
            
            # 使用 in_context_learning_s2s 方法
            text_output = model.in_context_learning_s2s(
                instruction,
                prompt_examples,
                input_audio,  # audio parameter
                max_new_tokens=8192,
                output_audio_path=str(output_audio_path)
            )
            
            results.append({
                'id': sample['id'],
                'input': input_audio,
                'output': str(output_audio_path),
                'expected': expected_output,
                'text': text,
                'success': True
            })
            
        except Exception as e:
            print(f"\n⚠️  Error processing {sample['id']}: {e}")
            results.append({
                'id': sample['id'],
                'input': input_audio,
                'success': False,
                'error': str(e)
            })
    
    # 儲存結果
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'shots': args.shots,
            'total_samples': len(test_samples_to_process),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # 總結
    successful = sum(1 for r in results if r['success'])
    print("\n" + "=" * 80)
    print(f"✅ 測試完成！")
    print(f"   成功: {successful}/{len(test_samples_to_process)}")
    print(f"   失敗: {len(test_samples_to_process) - successful}/{len(test_samples_to_process)}")
    print(f"   結果已儲存至: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ICL Testing on Optical Dataset")
    
    parser.add_argument('--test-split', type=str, required=True,
                        help='測試集 JSON 檔案')
    parser.add_argument('--shots', type=int, default=5,
                        help='Few-shot 數量')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='測試樣本數（從測試集前面選取）')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='輸出目錄')
    
    parser.add_argument('--model-path', type=str,
                        default='./models/MiMo-Audio-7B-Base',
                        help='MiMo-Audio 模型路徑')
    parser.add_argument('--tokenizer-path', type=str,
                        default='./models/MiMo-Audio-Tokenizer',
                        help='Tokenizer 路徑')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    test_icl(args)


if __name__ == '__main__':
    main()
