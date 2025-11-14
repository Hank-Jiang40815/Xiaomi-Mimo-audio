#!/usr/bin/env python3
"""
簡化的 Inference 測試
直接測試 MiMoAudioTokenizer 的 encode-decode
"""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.mimo_audio_tokenizer import MiMoAudioTokenizer

def test_basic_encode_decode():
    """測試基本的 encode-decode 流程（無 LoRA）"""
    
    device = 'cuda'
    tokenizer_path = 'models/MiMo-Audio-Tokenizer'
    input_audio = 'examples/optical/mix/boy1_WOLDV_050.wav'
    output_audio = 'outputs/test_inference/basic_test.wav'
    
    print("="*80)
    print("基本 Encode-Decode 測試")
    print("="*80)
    
    # 載入 tokenizer
    print(f"\n📦 載入 tokenizer: {tokenizer_path}")
    tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    tokenizer = tokenizer.to(device).to(torch.bfloat16)
    tokenizer.eval()
    
    # 載入音訊
    print(f"\n🎵 載入音訊: {input_audio}")
    waveform, sr = torchaudio.load(input_audio)
    
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"   Waveform shape: {waveform.shape}")
    print(f"   Duration: {waveform.shape[1] / 24000:.2f}s")
    
    # 轉換為 mel
    print("\n🔄 轉換為 Mel Spectrogram...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=128
    ).to(device)
    
    mel = mel_transform(waveform.to(device))
    print(f"   Mel shape (raw): {mel.shape}")  # [1, 128, T]
    
    # **關鍵發現：MimoAudio 使用的是 [T, 128] 格式！**
    mel = torch.log(torch.clip(mel, min=1e-7)).squeeze(0).transpose(0, 1)  # [T, 128]
    print(f"   Mel shape (processed): {mel.shape}")
    
    # **使用 MimoAudio 的方式：分段編碼**
    print("\n🔍 使用 MimoAudio 的編碼方式...")
    
    with torch.no_grad():
        input_len = mel.size(0)
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)
        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)
        
        print(f"   Input length: {input_len}")
        print(f"   Segments: {input_len_seg}")
        
        # 使用 AudioEncoder.encode（packed 格式）
        codes, output_length = tokenizer.encoder.encode(
            input_features=mel.to(torch.bfloat16),
            input_lens=torch.tensor(input_len_seg, device=device),
            return_codes_only=True
        )
        
        print(f"   ✅ 成功！")
        print(f"   Codes shape: {codes.shape}")
    
    # Decode
    print("\n🔄 Decode...")
    with torch.no_grad():
        reconstructed = tokenizer.decode(codes)
        print(f"   Reconstructed shape: {reconstructed.shape}")
    
    # 保存
    print(f"\n💾 保存到: {output_audio}")
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        output_audio,
        reconstructed.squeeze(0).cpu().float(),
        24000
    )
    
    print("\n✅ 測試完成！")
    print(f"   原始: {input_audio}")
    print(f"   重建: {output_audio}")

if __name__ == '__main__':
    test_basic_encode_decode()
