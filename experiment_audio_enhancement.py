# Copyright 2025 Xiaomi Corporation.
# Audio Enhancement Experiment using In-Context Learning
import os
from src.mimo_audio.mimo_audio import MimoAudio

model_path = "models/MiMo-Audio-7B-Base"
tokenizer_path = "models/MiMo-Audio-Tokenizer"

print("Loading MiMo-Audio Base model for audio enhancement experiment...")
model = MimoAudio(model_path, tokenizer_path)

# Experiment: Audio Enhancement / Denoising using In-Context Learning
# Task: Convert low-quality (LDV) audio to high-quality (clean) audio

instruction = "Enhance the audio quality and remove noise from the input speech. IMPORTANT: You MUST preserve the exact original speech content and transcript. The speech content is exactly 10 Chinese characters. Only improve the audio quality, do not change any words."

# Input: Low-quality audio that we want to enhance
# Use denoised optical dataset: mix=noisy, spk1=clean (denoised)
test_audio_files = [
    "examples/optical_denoised/mix/boy1_WOLDVlean_041.wav",
    "examples/optical_denoised/mix/boy1_WOLDVlean_042.wav",
    "examples/optical_denoised/mix/boy1_WOLDVlean_043.wav"
]

# Few-shot examples: Demonstrate LDV → clean transformation
# Note: We only have 2 example pairs, ideally we'd want 3-5 pairs
prompt_examples = [
    {
        "input_audio": "examples/optical_denoised/mix/boy1_WOLDVlean_001.wav",
        "output_audio": "examples/optical_denoised/spk1/boy1_papercup_clean_001.wav",
        "output_transcription": "這學期學校有書法比賽.",
    },
    {
        "input_audio": "examples/optical_denoised/mix/boy1_WOLDVlean_002.wav",
        "output_audio": "examples/optical_denoised/spk1/boy1_papercup_clean_002.wav",
        "output_transcription": "公司接到一份國外訂單.",
    },
    {
        "input_audio": "examples/optical_denoised/mix/boy1_WOLDVlean_003.wav",
        "output_audio": "examples/optical_denoised/spk1/boy1_papercup_clean_003.wav",
        "output_transcription": "他在禮堂主持開幕典禮.",
    },
    {
        "input_audio": "examples/optical_denoised/mix/boy1_WOLDVlean_004.wav",
        "output_audio": "examples/optical_denoised/spk1/boy1_papercup_clean_004.wav",
        "output_transcription": "這家書店今天正式營業.",
    },
    {
        "input_audio": "examples/optical_denoised/mix/boy1_WOLDVlean_005.wav",
        "output_audio": "examples/optical_denoised/spk1/boy1_papercup_clean_005.wav",
        "output_transcription": "今年夏天他剃了個光頭.",
    },
]

# Process each test audio file
for input_audio in test_audio_files:
    # Generate output filename based on test audio file
    # Add experiment tag to avoid overwriting previous results
    test_file_name = os.path.basename(input_audio).replace('.wav', '')
    output_audio_path = f"examples/audio_enhancement_{test_file_name}_5shot_strategy5_result.wav"

    print(f"\n{'='*60}")
    print("Audio Enhancement Experiment")
    print(f"{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"Input (to enhance): {input_audio}")
    print(f"Number of examples: {len(prompt_examples)}")
    print(f"Output will be saved to: {output_audio_path}")
    print(f"{'='*60}\n")

    try:
        text_channel_output = model.in_context_learning_s2s(
            instruction, 
            prompt_examples, 
            input_audio, 
            max_new_tokens=8192, 
            output_audio_path=output_audio_path
        )
        
        print(f"\n{'='*60}")
        print("Experiment completed successfully!")
        print(f"{'='*60}")
        print(f"Text channel output: {text_channel_output}")
        print(f"Enhanced audio saved to: {output_audio_path}")
        print(f"\nTo listen to the results:")
        print(f"  Original (LDV):  {input_audio}")
        print(f"  Enhanced:        {output_audio_path}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("Experiment failed!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Continuing to next test file...\n")
        continue

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}")
