# Copyright 2025 Xiaomi Corporation.
# Audio Enhancement Experiment using In-Context Learning
from src.mimo_audio.mimo_audio import MimoAudio

model_path = "models/MiMo-Audio-7B-Base"
tokenizer_path = "models/MiMo-Audio-Tokenizer"

print("Loading MiMo-Audio Base model for audio enhancement experiment...")
model = MimoAudio(model_path, tokenizer_path)

# Experiment: Audio Enhancement / Denoising using In-Context Learning
# Task: Convert low-quality (LDV) audio to high-quality (clean) audio

instruction = "Enhance the audio quality and remove noise from the input speech."

# Input: Low-quality audio that we want to enhance
input_audio = "examples/boy1_papercup_LDV_001.wav"

# Few-shot examples: Demonstrate LDV → clean transformation
# Note: We only have 2 example pairs, ideally we'd want 3-5 pairs
prompt_examples = [
    {
        "input_audio": "examples/boy1_papercup_LDV_002.wav",
        "output_audio": "examples/boy1_papercup_clean_002.wav",
        "output_transcription": "公司接到一份國外訂單.",  # LDV_002/clean_002 的內容
    },
]

output_audio_path = "examples/audio_enhancement_result.wav"

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
    print(f"  Reference clean: examples/boy1_papercup_clean_001.wav")
    
except Exception as e:
    print(f"\n{'='*60}")
    print("Experiment failed!")
    print(f"{'='*60}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
