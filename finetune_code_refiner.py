#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune MiMo-Audio code refinement module on top of frozen encoder+quantizer+decoder.

核心想法：
- 凍結 MiMo-Audio-Tokenizer 的 encoder/quantizer/decoder
- 使用小型 Transformer 對 noisy codes 做 refinement，讓其接近 clean codes
- Loss 以 code-level cross-entropy 為主，可選加上波形域約束（暫不實作）
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.mimo_audio_tokenizer import MiMoAudioTokenizer


class OpticalCodeDataset(Dataset):
    """Optical noisy/clean waveform pairs -> lazy-encoded codes via frozen tokenizer."""

    def __init__(self, manifest_dir: str, split: str = "train", sample_rate: int = 24000, max_length: float = 10.0):
        self.manifest_dir = Path(manifest_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)

        manifest_path = self.manifest_dir / f"{split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.data = manifest["samples"] if isinstance(manifest, dict) and "samples" in manifest else manifest
        if not self.data:
            raise ValueError(f"No samples in manifest: {manifest_path}")

        logger.info("Loaded %d samples from %s", len(self.data), manifest_path)

    def __len__(self):
        return len(self.data)

    def _load_waveform(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        num_samples = wav.shape[1]
        if num_samples > self.max_samples:
            wav = wav[:, : self.max_samples]
        elif num_samples < self.max_samples:
            pad = self.max_samples - num_samples
            wav = F.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        item = self.data[idx]
        noisy_path = str(Path(item["noisy_file"]))
        clean_path = str(Path(item["clean_file"]))
        noisy_wav = self._load_waveform(noisy_path)
        clean_wav = self._load_waveform(clean_path)
        return {
            "noisy_waveform": noisy_wav,
            "clean_waveform": clean_wav,
            "id": item.get("id", str(idx)),
        }


class CodeRefiner(nn.Module):
    """Small Transformer-based refiner operating on discrete code sequences."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model

        self.embed = nn.Embedding(codebook_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, codebook_size)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, T] int64
        Returns:
            logits: [B, T, codebook_size]
        """
        x = self.embed(codes)  # [B, T, d_model]
        x = self.encoder(x)
        logits = self.out_proj(x)
        return logits


def encode_waveforms_to_codes(tokenizer: MiMoAudioTokenizer, waveforms: torch.Tensor, device: torch.device):
    """
    將 batch waveform 透過 frozen encoder+quantizer 轉為 codes。

    Args:
        tokenizer: MiMoAudioTokenizer
        waveforms: [B, 1, samples] on device
    Returns:
        codes: [n_q, B, T'] int64
    """
    with torch.no_grad():
        # waveform -> Mel Spectrogram（與 encoder 設定一致）
        cfg = tokenizer.config
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sampling_rate,
            n_fft=1024,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=0.0,
            f_max=float(cfg.sampling_rate // 2),
        ).to(device)
        # waveforms: [B, 1, S] -> [B, n_mels, T]
        mels = mel_transform(waveforms).squeeze(1)
        B, n_mels, T = mels.shape
        input_lens = torch.full((B,), T, device=device, dtype=torch.long)

        # 使用 tokenizer.encoder.get_features + quantizer 取得 codes
        encoder = tokenizer.encoder
        quantizer = encoder.quantizer
        output_length = encoder.get_output_length(input_lens)
        hidden, _, _, _, _, _ = encoder.get_features(
            input_features=mels,
            output_length=output_length,
        )
        flat = hidden.reshape(-1, hidden.shape[-1])
        _, codes, _, _ = quantizer(flat)
        return codes


def collate_codes(batch, tokenizer: MiMoAudioTokenizer, device: torch.device):
    noisy_wavs = torch.stack([b["noisy_waveform"] for b in batch], dim=0).to(device)  # [B, 1, S]
    clean_wavs = torch.stack([b["clean_waveform"] for b in batch], dim=0).to(device)
    noisy_codes = encode_waveforms_to_codes(tokenizer, noisy_wavs, device)
    clean_codes = encode_waveforms_to_codes(tokenizer, clean_wavs, device)
    return noisy_codes, clean_codes


def train_one_epoch(
    tokenizer: MiMoAudioTokenizer,
    refiner: CodeRefiner,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    tokenizer.eval()
    refiner.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        noisy_codes, clean_codes = collate_codes(batch, tokenizer, device)
        # 目前只處理第一個 quantizer 層，簡化實驗
        noisy = noisy_codes[0].to(device)  # [T_flat]
        clean = clean_codes[0].to(device)
        # 假設所有樣本展平成同一長度，這裡直接視為 batch=1 序列
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)

        logits = refiner(noisy)  # [1, T, codebook_size]
        loss = F.cross_entropy(logits.view(-1, refiner.codebook_size), clean.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    tokenizer: MiMoAudioTokenizer,
    refiner: CodeRefiner,
    dataloader: DataLoader,
    device: torch.device,
):
    tokenizer.eval()
    refiner.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Validation"):
        noisy_codes, clean_codes = collate_codes(batch, tokenizer, device)
        noisy = noisy_codes[0].to(device).unsqueeze(0)
        clean = clean_codes[0].to(device).unsqueeze(0)
        logits = refiner(noisy)
        loss = F.cross_entropy(logits.view(-1, refiner.codebook_size), clean.view(-1))
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/splits/finetune_optical")
    parser.add_argument("--tokenizer-path", type=str, default="./models/MiMo-Audio-Tokenizer")
    parser.add_argument("--output-dir", type=str, default="./outputs/code_refiner_optical")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--projector-d-model", type=int, default=256)
    parser.add_argument("--projector-nhead", type=int, default=4)
    parser.add_argument("--projector-layers", type=int, default=2)
    parser.add_argument("--projector-ff", type=int, default=1024)
    parser.add_argument("--projector-dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = MiMoAudioTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer = tokenizer.to(device).eval()

    # 取得第一個 quantizer 的 codebook size 作為 refiner 的 vocab
    q = tokenizer.encoder.quantizer.vq.layers[0]
    codebook_size = q._codebook.embed.shape[0]
    logger.info("CodeRefiner vocab size (first quantizer): %d", codebook_size)

    refiner = CodeRefiner(
        codebook_size=codebook_size,
        d_model=args.projector_d_model,
        nhead=args.projector_nhead,
        num_layers=args.projector_layers,
        dim_feedforward=args.projector_ff,
        dropout=args.projector_dropout,
    ).to(device)

    train_ds = OpticalCodeDataset(args.data_dir, split="train")
    val_ds = OpticalCodeDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)
        train_loss = train_one_epoch(tokenizer, refiner, train_loader, optimizer, device, epoch)
        val_loss = validate(tokenizer, refiner, val_loader, device)
        logger.info("Train Loss: %.4f", train_loss)
        logger.info("Val Loss:   %.4f", val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        ckpt = {
            "epoch": epoch,
            "state_dict": refiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, output_dir / "best_model.pt")
            logger.info("Saved best model (val_loss=%.4f)", val_loss)
        if epoch % 10 == 0:
            torch.save(ckpt, output_dir / f"checkpoint_epoch_{epoch}.pt")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training completed. Best Val Loss: %.4f", best_val)


if __name__ == "__main__":
    main()
