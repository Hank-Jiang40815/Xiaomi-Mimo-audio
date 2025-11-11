#!/usr/bin/env python3
"""
重現 exp(D2) 實驗
Commit: 10c7149 - Test x65 and x70 variants with 40-shot ICL
"""

import os
import sys
from src.mimo_audio.mimo_audio import MimoAudio

# 配置
model_path = "models/MiMo-Audio-7B-Base"
tokenizer_path = "models/MiMo-Audio-Tokenizer"

print("╔═══════════════════════════════════════════════════════════════╗")
print("║     重現 exp(D2): x65/x70 噪聲級別測試 (40-shot ICL)          ║")
print("╚═══════════════════════════════════════════════════════════════╝")
print()

# 載入模型
print("🔄 載入 MiMo-Audio-7B-Base 模型...")
model = MimoAudio(model_path, tokenizer_path)
print("✅ 模型載入成功\n")

# Instruction
instruction = "Enhance the audio quality and remove noise from the input speech. IMPORTANT: You MUST preserve the exact original speech content and transcript. Only improve the audio quality, do not change any words."

# 測試檔案
test_audio_files = [
    # x65 series (較低噪聲級別)
    "examples/ldv/mix/boy1_papercup_LDV_x65_289.wav",
    "examples/ldv/mix/boy1_papercup_LDV_x65_290.wav",
    "examples/ldv/mix/boy1_papercup_LDV_x65_291.wav",
    # x70 series (較高噪聲級別)
    "examples/ldv/mix/boy1_papercup_LDV_x70_299.wav",
    "examples/ldv/mix/boy1_papercup_LDV_x70_300.wav",
    "examples/ldv/mix/boy1_papercup_LDV_x70_301.wav",
]

# 40-shot 訓練範例 (001-040)
print("📊 準備 40-shot 訓練範例...")
prompt_examples = []
for i in range(1, 41):
    prompt_examples.append({
        "input_audio": f"examples/ldv/mix/boy1_papercup_LDV_{i:03d}.wav",
        "output_audio": f"examples/ldv/spk/boy1_papercup_clean_{i:03d}.wav",
        "output_transcription": "",  # 可選
    })
print(f"✅ 準備了 {len(prompt_examples)} 個訓練範例\n")

# 執行推論
success_count = 0
failed_count = 0

for idx, input_audio in enumerate(test_audio_files, 1):
    # 生成輸出檔名
    test_file_name = os.path.basename(input_audio).replace('.wav', '').replace('boy1_papercup_', '')
    output_audio_path = f"examples/audio_enhancement_{test_file_name}_ldv_40shot_result.wav"
    
    print(f"\n{'='*70}")
    print(f"[{idx}/{len(test_audio_files)}] 處理: {os.path.basename(input_audio)}")
    print(f"{'='*70}")
    print(f"📥 輸入: {input_audio}")
    print(f"📤 輸出: {output_audio_path}")
    print()
    
    try:
        text_channel_output = model.in_context_learning_s2s(
            instruction,
            prompt_examples,
            input_audio,
            max_new_tokens=8192,
            output_audio_path=output_audio_path
        )
        
        # 檢查檔案大小
        file_size = os.path.getsize(output_audio_path) / 1024  # KB
        
        print(f"\n✅ 成功！")
        print(f"   文字輸出: {text_channel_output}")
        print(f"   檔案大小: {file_size:.1f} KB")
        print(f"   儲存位置: {output_audio_path}")
        
        success_count += 1
        
    except Exception as e:
        print(f"\n❌ 失敗！")
        print(f"   錯誤: {e}")
        import traceback
        traceback.print_exc()
        failed_count += 1
        print(f"\n⏭️  繼續處理下一個檔案...")

# 最終統計
print(f"\n\n{'='*70}")
print(f"實驗完成！")
print(f"{'='*70}")
print(f"✅ 成功: {success_count}/{len(test_audio_files)}")
print(f"❌ 失敗: {failed_count}/{len(test_audio_files)}")
print(f"📊 成功率: {success_count/len(test_audio_files)*100:.1f}%")
print(f"{'='*70}\n")

# 顯示結果檔案
if success_count > 0:
    print("📁 生成的檔案:")
    for input_audio in test_audio_files:
        test_file_name = os.path.basename(input_audio).replace('.wav', '').replace('boy1_papercup_', '')
        output_path = f"examples/audio_enhancement_{test_file_name}_ldv_40shot_result.wav"
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024
            print(f"   ✓ {output_path} ({size:.1f} KB)")

print("\n實驗重現完成！🎉")
