#!/usr/bin/env python3
"""
Optical Dataset Audio Transcription Tool
使用 Whisper 自動轉錄乾淨音檔 (spk1/) 並生成 JSON 格式的轉錄結果
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import whisper
import warnings

# 忽略 FP16 警告
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


def transcribe_audio_files(
    audio_dir: str,
    output_json: str,
    model_size: str = "base",
    batch_size: int = 1,
    language: str = "zh",
    resume: bool = False,
):
    """
    批次轉錄音檔並儲存為 JSON 格式
    
    Args:
        audio_dir: 音檔目錄路徑
        output_json: 輸出 JSON 檔案路徑
        model_size: Whisper 模型大小 (tiny/base/small/medium/large)
        batch_size: 批次處理大小 (目前只支援 1)
        language: 音檔語言 (zh=中文, en=英文)
        resume: 是否從上次中斷處繼續
    """
    
    audio_dir = Path(audio_dir)
    output_json = Path(output_json)
    
    # 確保輸出目錄存在
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # 載入已有的轉錄結果（如果要繼續）
    transcriptions = {}
    if resume and output_json.exists():
        print(f"📂 載入現有轉錄結果: {output_json}")
        with open(output_json, 'r', encoding='utf-8') as f:
            transcriptions = json.load(f)
        print(f"   已有 {len(transcriptions)} 筆轉錄記錄")
    
    # 獲取所有音檔
    audio_files = sorted(audio_dir.glob("*.wav"))
    total_files = len(audio_files)
    
    if total_files == 0:
        print(f"❌ 在 {audio_dir} 找不到任何 .wav 檔案")
        return
    
    # 過濾已轉錄的檔案
    if resume:
        audio_files = [f for f in audio_files if f.name not in transcriptions]
        print(f"   剩餘 {len(audio_files)} 個檔案需要轉錄")
    
    print(f"\n🎤 準備轉錄 {len(audio_files)} 個音檔")
    print(f"   音檔目錄: {audio_dir}")
    print(f"   輸出檔案: {output_json}")
    print(f"   Whisper 模型: {model_size}")
    print(f"   語言: {language}")
    
    # 載入 Whisper 模型
    print(f"\n📥 載入 Whisper {model_size} 模型...")
    model = whisper.load_model(model_size)
    print("✅ 模型載入完成")
    
    # 開始轉錄
    print(f"\n{'='*60}")
    print("開始轉錄...")
    print(f"{'='*60}\n")
    
    success_count = 0
    error_count = 0
    
    for audio_file in tqdm(audio_files, desc="轉錄進度", unit="檔案"):
        try:
            # 轉錄音檔
            result = model.transcribe(
                str(audio_file),
                language=language,
                fp16=False,  # CPU 使用 FP32
            )
            
            # 儲存結果
            transcriptions[audio_file.name] = {
                "text": result["text"].strip(),
                "language": result["language"],
            }
            
            success_count += 1
            
            # 定期儲存（每 100 筆）
            if success_count % 100 == 0:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(transcriptions, f, ensure_ascii=False, indent=2)
                tqdm.write(f"💾 已儲存 {success_count} 筆轉錄結果")
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"❌ 轉錄失敗: {audio_file.name} - {e}")
            continue
    
    # 最終儲存
    print(f"\n💾 儲存最終轉錄結果...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    
    # 統計資訊
    print(f"\n{'='*60}")
    print("轉錄完成！")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count} 個檔案")
    print(f"❌ 失敗: {error_count} 個檔案")
    print(f"📊 總計: {total_files} 個檔案")
    print(f"💾 結果已儲存至: {output_json}")
    print(f"   檔案大小: {output_json.stat().st_size / 1024:.1f} KB")
    
    # 顯示幾個範例
    if transcriptions:
        print(f"\n📝 轉錄範例:")
        for i, (filename, data) in enumerate(list(transcriptions.items())[:3], 1):
            print(f"   {i}. {filename}")
            print(f"      文字: {data['text'][:60]}...")
            print(f"      語言: {data['language']}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 Whisper 批次轉錄 optical 資料集音檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 基本使用（使用 base 模型）
  python transcribe_optical.py

  # 使用更大的模型以獲得更好的準確度
  python transcribe_optical.py --model-size small

  # 從中斷處繼續
  python transcribe_optical.py --resume

  # 自訂路徑
  python transcribe_optical.py \\
    --audio-dir examples/optical/spk1 \\
    --output data/transcriptions/optical_transcriptions.json

模型大小說明:
  tiny   - 最快，準確度較低（~1GB RAM）
  base   - 平衡選擇（~1GB RAM）
  small  - 更準確（~2GB RAM）
  medium - 很準確（~5GB RAM）
  large  - 最準確，最慢（~10GB RAM）
        """
    )
    
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="examples/optical/spk1",
        help="音檔目錄路徑（預設: examples/optical/spk1）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/transcriptions/optical_transcriptions.json",
        help="輸出 JSON 檔案路徑（預設: data/transcriptions/optical_transcriptions.json）"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper 模型大小（預設: base）"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="音檔語言代碼（zh=中文, en=英文, 預設: zh）"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從上次中斷處繼續轉錄"
    )
    
    args = parser.parse_args()
    
    transcribe_audio_files(
        audio_dir=args.audio_dir,
        output_json=args.output,
        model_size=args.model_size,
        language=args.language,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
