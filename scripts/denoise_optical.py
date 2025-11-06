#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import soundfile as sf  # preferred for reading/writing
    _HAVE_SF = True
except Exception:
    sf = None
    _HAVE_SF = False
    from scipy.io import wavfile as wav

from scipy.signal import butter, sosfiltfilt

try:
    import noisereduce as nr
except Exception as e:  # pragma: no cover
    print("[ERROR] 缺少 noisereduce 套件，請先安裝：pip install noisereduce", file=sys.stderr)
    raise


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak == 0 or not np.isfinite(peak):
        return audio
    target_linear = 10 ** (target_dbfs / 20.0)
    out = audio * (target_linear / peak)
    return np.clip(out, -1.0, 1.0)


def _read_audio(path: Path):
    if _HAVE_SF:
        data, sr = sf.read(str(path), always_2d=False)
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float64) / max_val
        else:
            data = data.astype(np.float64)
        return data, sr
    # Fallback WAV-only
    sr, data = wav.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float64)
    return data, sr


def _write_audio(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_16"):
    if _HAVE_SF:
        sf.write(str(path), data, sr, subtype=subtype)
    else:
        out = np.clip(data, -1.0, 1.0)
        wav.write(str(path), sr, (out * 32767.0).astype(np.int16))


def apply_filters(y: np.ndarray, sr: int, hp: float | None, lp: float | None) -> np.ndarray:
    out = y
    nyq = sr / 2.0
    if hp is not None and hp > 0 and hp < nyq:
        w = hp / nyq
        sos = butter(2, w, btype='highpass', output='sos')
        out = sosfiltfilt(sos, out).astype(out.dtype, copy=False)
    if lp is not None and lp > 0 and lp < nyq:
        w = lp / nyq
        sos = butter(4, w, btype='lowpass', output='sos')
        out = sosfiltfilt(sos, out).astype(out.dtype, copy=False)
    return out


def denoise_once(y: np.ndarray, sr: int,
                 stationary: bool,
                 prop_decrease: float,
                 time_mask_smooth_ms: float,
                 freq_mask_smooth_hz: float) -> np.ndarray:
    # noisereduce 期望單聲道；多聲道逐軌處理
    if y.ndim == 1:
        return nr.reduce_noise(
            y=y, sr=sr, stationary=stationary, prop_decrease=prop_decrease,
            time_mask_smooth_ms=time_mask_smooth_ms, freq_mask_smooth_hz=freq_mask_smooth_hz,
        )
    else:
        ch = y.shape[1]
        outs = []
        for i in range(ch):
            outs.append(nr.reduce_noise(
                y=y[:, i], sr=sr, stationary=stationary, prop_decrease=prop_decrease,
                time_mask_smooth_ms=time_mask_smooth_ms, freq_mask_smooth_hz=freq_mask_smooth_hz,
            ))
        return np.stack(outs, axis=1)


def process_file(src: Path, dst: Path,
                 hp: float | None, lp: float | None,
                 stationary: bool, prop_decrease: float,
                 time_mask_smooth_ms: float, freq_mask_smooth_hz: float,
                 target_dbfs: float,
                 subtype: str, dry_run: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    y, sr = _read_audio(src)

    # 頻帶整形（輕量）
    y_f = apply_filters(y, sr, hp=hp, lp=lp)

    # 降噪（非平穩，避免過度產生顆粒）
    y_dn = denoise_once(y_f, sr,
                        stationary=stationary,
                        prop_decrease=prop_decrease,
                        time_mask_smooth_ms=time_mask_smooth_ms,
                        freq_mask_smooth_hz=freq_mask_smooth_hz)

    # 輕量峰值正規化
    y_out = peak_normalize(y_dn, target_dbfs=target_dbfs)

    if dry_run:
        print(f"[DRY-RUN] {src} -> {dst} | sr={sr}")
        return

    _write_audio(dst, y_out, sr, subtype=subtype)
    print(f"[WROTE] {dst}")


def collect_audio_files(root: Path, exts=(".wav",)):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser(description="Denoise optical_normalized audio into optical_denoised using noisereduce.")
    ap.add_argument('--src-root', type=Path, default=Path('examples/optical_normalized'))
    ap.add_argument('--out-root', type=Path, default=Path('examples/optical_denoised'))
    ap.add_argument('--subdirs', nargs='*', default=['mix', 'spk1'])
    ap.add_argument('--hp', type=float, default=90.0, help='High-pass cutoff Hz (None/0 to disable)')
    ap.add_argument('--lp', type=float, default=8500.0, help='Low-pass cutoff Hz (None/0 to disable)')
    ap.add_argument('--stationary', action='store_true', help='Use stationary noise reduction (default: non-stationary)')
    ap.add_argument('--prop-decrease', type=float, default=0.85)
    ap.add_argument('--time-mask-smooth-ms', type=float, default=64.0)
    ap.add_argument('--freq-mask-smooth-hz', type=float, default=150.0)
    ap.add_argument('--target-dbfs', type=float, default=-1.0)
    ap.add_argument('--subtype', type=str, default='PCM_16')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true')

    args = ap.parse_args()

    if not args.src_root.exists():
        print(f"[ERR] Source root not found: {args.src_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    skipped = 0

    for sub in args.subdirs:
        sdir = args.src_root / sub
        if not sdir.exists():
            print(f"[WARN] missing subdir: {sdir}")
            continue
        odir = args.out_root / sub

        for src in sorted(collect_audio_files(sdir)):
            rel = src.relative_to(sdir)
            dst = odir / rel
            if dst.exists() and not args.overwrite and not args.dry_run:
                print(f"[SKIP] Exists: {dst}")
                skipped += 1
                continue
            process_file(src, dst,
                         hp=(None if not args.hp or args.hp <= 0 else args.hp),
                         lp=(None if not args.lp or args.lp <= 0 else args.lp),
                         stationary=args.stationary,
                         prop_decrease=args.prop_decrease,
                         time_mask_smooth_ms=args.time_mask_smooth_ms,
                         freq_mask_smooth_hz=args.freq_mask_smooth_hz,
                         target_dbfs=args.target_dbfs,
                         subtype=args.subtype,
                         dry_run=args.dry_run)
            total += 1

    print(f"Done. processed={total}, skipped={skipped}, out_root={args.out_root}")


if __name__ == '__main__':
    main()

