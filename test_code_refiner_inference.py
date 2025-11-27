#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torchaudio

from src.mimo_audio_tokenizer import MiMoAudioTokenizer
from finetune_code_refiner import CodeRefiner, OpticalCodeDataset, encode_waveforms_to_codes


def load_wave(path: str, sr: int) -> torch.Tensor:
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer-path", type=str, default="./models/MiMo-Audio-Tokenizer")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--num-quantizer-layers", type=int, default=1, help="How many RVQ layers to refine (from layer 0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer = tokenizer.to(device).to(torch.bfloat16).eval()

    # 建立 refiner 並載入權重
    q = tokenizer.encoder.quantizer.vq.layers[0]
    codebook_size = q._codebook.embed.shape[0]
    ckpt = torch.load(args.checkpoint, map_location=device)
    refiner = CodeRefiner(
        codebook_size=codebook_size,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
    ).to(device)
    refiner.load_state_dict(ckpt["state_dict"])
    refiner.eval()

    sr = tokenizer.config.sampling_rate
    wav = load_wave(args.input, sr).to(device)
    wav = wav.unsqueeze(0)  # [1, 1, S]

    with torch.no_grad():
        codes = encode_waveforms_to_codes(tokenizer, wav, device)  # [n_q, T]
        refined_codes = codes.clone()
        n_q = codes.shape[0]
        layers = min(args.num_quantizer_layers, n_q)
        for q in range(layers):
            noisy = codes[q].unsqueeze(0).to(device)  # [1, T]
            logits = refiner(noisy)
            refined = logits.argmax(dim=-1)  # [1, T]
            refined_codes[q] = refined.squeeze(0)

        # decode back to waveform
        hidden = tokenizer.encoder.decode_vq(refined_codes)
        out = tokenizer.decoder(
            hidden,
            torch.tensor([hidden.size(0)], device=hidden.device),
        )[0]
        out_wav = out.float().cpu()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), out_wav, sr)
    print(f"Saved refined audio to {out_path}")


if __name__ == "__main__":
    main()
