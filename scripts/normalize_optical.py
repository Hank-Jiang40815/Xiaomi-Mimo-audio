#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
try:
    import soundfile as sf  # preferred: supports many formats
    _HAVE_SF = True
except Exception:  # pragma: no cover
    sf = None
    _HAVE_SF = False
    from scipy.io import wavfile as wav  # fallback for .wav only


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Peak-normalize audio to a target dBFS level.

    Args:
        audio: np.ndarray, shape (n,) or (n, channels), float32/float64 in [-1, 1]
        target_dbfs: target peak level in dBFS (e.g., -1.0 dB)

    Returns:
        Normalized audio with the same shape and dtype as input (float64 preserved).
    """
    if audio.size == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak == 0 or not np.isfinite(peak):
        return audio

    target_linear = 10 ** (target_dbfs / 20.0)
    scale = target_linear / peak
    out = audio * scale
    # Safety clip to avoid any rounding overflow when writing PCM
    out = np.clip(out, -1.0, 1.0)
    return out


def _read_audio(src: Path):
    if _HAVE_SF:
        data, sr = sf.read(str(src), always_2d=False)
        return data, sr
    # Fallback: WAV-only
    if src.suffix.lower() != ".wav":
        raise RuntimeError("SoundFile not available; only .wav supported in fallback")
    sr, data = wav.read(str(src))
    return data, sr


def _write_audio(dst: Path, data: np.ndarray, sr: int, subtype: str):
    if _HAVE_SF:
        sf.write(str(dst), data, sr, subtype=subtype)
    else:
        # SciPy fallback: write PCM16 WAV to match default subtype intent
        out = np.clip(data, -1.0, 1.0)
        out_i16 = (out * 32767.0).astype(np.int16)
        wav.write(str(dst), sr, out_i16)


def process_file(src: Path, dst: Path, target_dbfs: float, subtype: str, dry_run: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data, sr = _read_audio(src)

    # Ensure float domain for normalization
    if np.issubdtype(data.dtype, np.integer):
        # Convert integers to float in [-1, 1]
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    norm = peak_normalize(data, target_dbfs)

    if dry_run:
        peak_before = float(np.max(np.abs(data))) if data.size else 0.0
        peak_after = float(np.max(np.abs(norm))) if norm.size else 0.0
        print(f"[DRY-RUN] {src} -> {dst} | sr={sr} | peak_before={peak_before:.6f} | peak_after={peak_after:.6f}")
        return

    _write_audio(dst, norm, sr, subtype=subtype)
    print(f"[WROTE] {dst}")


def collect_audio_files(root: Path, exts=(".wav", ".flac", ".mp3", ".ogg")):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Normalize audio files under examples/optical subfolders.")
    parser.add_argument("--src-root", type=Path, default=Path("examples/optical"), help="Source root containing subfolders (default: examples/optical)")
    parser.add_argument("--out-root", type=Path, default=Path("examples/optical_normalized"), help="Output root for normalized audio")
    parser.add_argument("--subdirs", nargs="*", default=["mix", "spk1"], help="Subdirectories under src-root to process")
    parser.add_argument("--target-dbfs", type=float, default=-1.0, help="Target peak level in dBFS (default: -1.0 dB)")
    parser.add_argument("--subtype", type=str, default="PCM_16", help="WAV subtype for output (e.g., PCM_16, PCM_24, FLOAT)")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized files")

    args = parser.parse_args()

    if not args.src_root.exists():
        print(f"Source root not found: {args.src_root}", file=sys.stderr)
        sys.exit(1)

    processed = 0
    skipped = 0

    for sub in args.subdirs:
        src_dir = args.src_root / sub
        if not src_dir.exists():
            print(f"[WARN] Subdir not found, skipping: {src_dir}")
            continue

        out_dir = args.out_root / sub

        for src_file in sorted(collect_audio_files(src_dir)):
            rel = src_file.relative_to(src_dir)
            dst_file = out_dir / rel
            if dst_file.exists() and not args.overwrite and not args.dry_run:
                print(f"[SKIP] Exists: {dst_file}")
                skipped += 1
                continue

            process_file(src_file, dst_file, args.target_dbfs, args.subtype, dry_run=args.dry_run)
            processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, out_root={args.out_root}")


if __name__ == "__main__":
    main()
