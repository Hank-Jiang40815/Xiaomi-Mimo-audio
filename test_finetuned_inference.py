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
    tokenizer = tokenizer.to(torch.bfloat16)  # Flash Attention 需求
    
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
    print(f"   載入 {len(checkpoint)} 個 LoRA 參數...")
    tokenizer.encoder.load_state_dict(checkpoint, strict=False)
    
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
    
    # 2. 轉換為 Mel Spectrogram（和訓練時一樣）
    print("\n2️⃣ 轉換為 Mel Spectrogram...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=240,
        n_mels=128
    ).to(device)
    
    waveform = waveform.to(device).to(torch.bfloat16)
    mel_spec = mel_transform(waveform)  # [1, 128, T]
    
    print(f"   Mel Spectrogram Shape: {mel_spec.shape}")
    
    # 3. 編碼（使用微調後的 encoder）
    print("\n3️⃣ 使用微調後的 Encoder 編碼...")
    
    with torch.no_grad():
        # 編碼到 latent space
        input_lens = torch.tensor([mel_spec.shape[2]], device=device)
        encoded = model.encode(mel_spec, input_lens=input_lens)
        
        print(f"   Encoded Shape: {encoded.shape}")
        
        # 解碼回 Mel Spectrogram
        print("\n4️⃣ 解碼回 Mel Spectrogram...")
        decoded_mel = model.decode(encoded, input_lens=input_lens)
        
        print(f"   Decoded Mel Shape: {decoded_mel.shape}")
        
        # 使用 Griffin-Lim 或 Vocoder 轉回波形
        # 這裡簡單使用 inverse mel scale
        print("\n5️⃣ 轉回波形...")
        # 注意: 這是簡化版本，實際應該使用訓練好的 vocoder
        decoded = decoded_mel  # 暫時返回 mel (後續可加入 vocoder)
    
    # 6. 儲存結果
    print("\n6️⃣ 儲存結果...")
    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 注意: decoded 現在是 mel spectrogram，需要轉回波形
    # 這裡暫時保存為 .pt 格式（mel spectrogram）
    print("   ⚠️  當前版本保存 Mel Spectrogram（需要 vocoder 才能轉回音訊）")
    output_pt = output_path.with_suffix('.pt')
    torch.save(decoded.cpu(), output_pt)
    
    file_size = output_path.stat().st_size / 1024  # KB
    
    print(f"\n✅ 增強完成！")
    print(f"   檔案: {output_audio_path}")
    print(f"   大小: {file_size:.1f} KB")
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
