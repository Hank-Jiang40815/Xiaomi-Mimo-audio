#!/usr/bin/env python3
"""
測試微調後 Encoder 的 Inference
載入 LoRA checkpoint 並對測試音訊進行增強
"""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.mimo_audio_tokenizer import MiMoAudioTokenizer

def load_finetuned_model(checkpoint_path, tokenizer_path, device='cuda'):
    """
    載入微調後的模型
    
    Args:
        checkpoint_path: LoRA checkpoint 路徑 (例如: checkpoint_epoch_2.pt)
        tokenizer_path: 基礎 tokenizer 路徑
        device: 運行設備
    """
    print("=" * 80)
    print("🔧 載入微調後的模型")
    print("=" * 80)
    
    # 1. 載入基礎 tokenizer  
    print(f"\n1️⃣ 載入基礎 tokenizer: {tokenizer_path}")
    tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    tokenizer = tokenizer.to(device)
    # 先不轉 bfloat16，等 LoRA 注入後再轉
    
    # 2. 載入 LoRA checkpoint
    print(f"\n2️⃣ 載入 LoRA checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 3. 應用 LoRA 權重到 encoder
    print("\n3️⃣ 應用 LoRA 權重...")
    
    # 重新注入 LoRA (和訓練時一樣)
    from finetune_encoder import inject_lora_to_encoder
    
    lora_config = {
        'rank': 32,
        'alpha': 64
    }
    
    inject_lora_to_encoder(
        tokenizer.encoder,
        rank=lora_config['rank'],
        alpha=lora_config['alpha']
    )
    
    # 載入 checkpoint 的 LoRA 權重
    print(f"   載入 {len(checkpoint['lora_state_dict'])} 個 LoRA 參數...")
    for name, param in tokenizer.encoder.named_parameters():
        if name in checkpoint['lora_state_dict']:
            param.data = checkpoint['lora_state_dict'][name].to(device)
    
    # **關鍵：在載入 LoRA 後，將整個 tokenizer 轉為 bfloat16**
    print("   ✅ 轉換整個模型為 bfloat16...")
    tokenizer = tokenizer.to(torch.bfloat16)
    
    print("\n✅ 模型載入完成！")
    print("=" * 80)
    
    return tokenizer


def enhance_audio(model, input_audio_path, output_audio_path, device='cuda'):
    """
    使用微調後的 encoder 增強音訊
    
    Args:
        model: 微調後的 tokenizer
        input_audio_path: 輸入噪聲音訊路徑
        output_audio_path: 輸出增強音訊路徑
        device: 運行設備
    """
    print("\n" + "=" * 80)
    print("🎵 音訊增強處理")
    print("=" * 80)
    print(f"📥 輸入: {input_audio_path}")
    print(f"📤 輸出: {output_audio_path}")
    print()
    
    # 1. 載入音訊
    print("1️⃣ 載入音訊...")
    waveform, sr = torchaudio.load(input_audio_path)
    
    # 轉換到 24kHz
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)
        sr = 24000
    
    # 轉換為 mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    print(f"   採樣率: {sr} Hz")
    print(f"   長度: {waveform.shape[1] / sr:.2f} 秒")
    print(f"   Waveform Shape: {waveform.shape}")
    
    # 2. 轉換為 Mel Spectrogram（使用 MimoAudio 的方式）
    print("\n2️⃣ 轉換為 Mel Spectrogram...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=128
    ).to(device)
    
    waveform_gpu = waveform.to(device)
    mel_raw = mel_transform(waveform_gpu)  # [1, 128, T]
    
    # **關鍵：使用 MimoAudio 的格式 [T, 128]**
    mel = torch.log(torch.clip(mel_raw, min=1e-7)).squeeze(0).transpose(0, 1)  # [T, 128]
    print(f"   Mel Spectrogram Shape: {mel.shape}")  # [T, 128]
    
    # 3. 編碼（使用微調後的 encoder）
    print("\n3️⃣ 使用微調後的 Encoder 編碼...")
    
    with torch.no_grad():
        # 使用 MimoAudio 的分段方式
        input_len = mel.size(0)
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)
        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)
        
        print(f"   Segments: {input_len_seg}")
        
        # 使用 AudioEncoder.encode（微調後的）
        codes, output_length = model.encoder.encode(
            input_features=mel.to(torch.bfloat16),
            input_lens=torch.tensor(input_len_seg, device=device),
            return_codes_only=True
        )
        
        print(f"   Codes Shape: {codes.shape}")  # [num_quantizers, T']
        
        # 4. 解碼回音訊波形
        print("\n4️⃣ 解碼回音訊波形...")
        # decode() 接受離散的 codes 並輸出波形
        reconstructed_audio = model.decode(codes)  # [1, 1, samples]
        
        print(f"   Reconstructed Audio Shape: {reconstructed_audio.shape}")
    
    # 5. 儲存結果
    print("\n5️⃣ 儲存結果...")
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存重建的音訊
    torchaudio.save(
        str(output_path),
        reconstructed_audio.squeeze(0).cpu().float(),  # [1, samples]
        24000
    )
    
    file_size = output_path.stat().st_size / 1024  # KB
    duration = reconstructed_audio.shape[-1] / 24000
    
    print(f"\n✅ 增強完成！")
    print(f"   檔案: {output_audio_path}")
    print(f"   大小: {file_size:.1f} KB")
    print(f"   時長: {duration:.2f} 秒")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="測試微調後 Encoder 的 Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='LoRA checkpoint 路徑')
    parser.add_argument('--tokenizer-path', type=str,
                        default='./models/MiMo-Audio-Tokenizer',
                        help='基礎 tokenizer 路徑')
    parser.add_argument('--input', type=str, required=True,
                        help='輸入噪聲音訊路徑')
    parser.add_argument('--output', type=str, required=True,
                        help='輸出增強音訊路徑')
    parser.add_argument('--device', type=str, default='cuda',
                        help='運行設備')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🧪 微調後 Encoder Inference 測試")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 載入模型
    model = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )
    
    # 增強音訊
    enhance_audio(
        model=model,
        input_audio_path=args.input,
        output_audio_path=args.output,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("✨ 測試完成！")
    print("=" * 80)
    print()
    print("🎧 播放指令:")
    print(f"   原始: ffplay -autoexit -nodisp {args.input}")
    print(f"   增強: ffplay -autoexit -nodisp {args.output}")
    print()


if __name__ == '__main__':
    main()
