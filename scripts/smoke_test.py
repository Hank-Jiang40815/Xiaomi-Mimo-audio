import argparse
import os
import sys
from pathlib import Path

import torch

from src.mimo_audio.mimo_audio import MimoAudio


def main():
    parser = argparse.ArgumentParser(description="MiMo-Audio GPU smoke test (no web UI)")
    parser.add_argument("--model_path", default=os.environ.get("MIMO_MODEL", "./models/MiMo-Audio-7B-Instruct"))
    parser.add_argument("--tokenizer_path", default=os.environ.get("MIMO_TOKENIZER", "./models/MiMo-Audio-Tokenizer"))
    parser.add_argument("--audio", default="./examples/spoken_dialogue_assistant_turn_1.wav", help="Path to an example input audio file (wav)")
    parser.add_argument("--out_dir", default="./outputs", help="Directory to save generated audio")
    args = parser.parse_args()

    print("=== MiMo-Audio Smoke Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    audio_path = args.audio
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Example audio: {audio_path}")

    # Load model
    mimo = MimoAudio(model_path, tokenizer_path)

    # 1) Audio Understanding on a real wav from repo
    if not os.path.exists(audio_path):
        print(f"ERROR: example audio not found at {audio_path}", file=sys.stderr)
        sys.exit(1)
    question = "請用一句話描述這段音檔的內容。"
    print("\n[Audio Understanding]")
    au_text = mimo.audio_understanding_sft(audio_path, question, thinking=False)
    print("Answer:", au_text)

    # 2) TTS simple generation
    print("\n[Text-to-Speech]")
    tts_text = "你好，這是一段測試語音。"
    tts_out = str(out_dir / "tts_sample.wav")
    text_channel = mimo.tts_sft(tts_text, tts_out, instruct=None, read_text_only=True)
    print("Text channel:", text_channel)
    print(f"Generated audio saved to: {tts_out}")

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()

