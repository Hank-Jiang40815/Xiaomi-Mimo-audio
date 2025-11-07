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
# Dataset: LDV (examples/ldv/) - Original LDV dataset for comparison with optical_denoised

instruction = "Enhance the audio quality and remove noise from the input speech. IMPORTANT: You MUST preserve the exact original speech content and transcript. Only improve the audio quality, do not change any words."

# Input: Low-quality audio that we want to enhance
# Use LDV dataset: mix=noisy (LDV), spk=clean
test_audio_files = [
    "examples/ldv/mix/boy1_papercup_LDV_041.wav",
    "examples/ldv/mix/boy1_papercup_LDV_042.wav",
    "examples/ldv/mix/boy1_papercup_LDV_043.wav"
]

# Few-shot examples: Demonstrate LDV → clean transformation
# Using 40-shot configuration (001-040) proven effective in previous experiments
prompt_examples = [
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_001.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_001.wav",
        "output_transcription": "這學期學校有書法比賽.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_002.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_002.wav",
        "output_transcription": "公司接到一份國外訂單.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_003.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_003.wav",
        "output_transcription": "他在禮堂主持開幕典禮.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_004.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_004.wav",
        "output_transcription": "這家書店今天正式營業.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_005.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_005.wav",
        "output_transcription": "今年夏天他剃了個光頭.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_006.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_006.wav",
        "output_transcription": "這群訪客都戴著識別證.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_007.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_007.wav",
        "output_transcription": "他聽到這個消息很傷心.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_008.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_008.wav",
        "output_transcription": "他在扭傷的腳上敷冰塊.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_009.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_009.wav",
        "output_transcription": "這兩個寺廟的香火很盛.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_010.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_010.wav",
        "output_transcription": "秘書在幫老闆撰寫文件.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_011.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_011.wav",
        "output_transcription": "我今年初一像爸爸拜年.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_012.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_012.wav",
        "output_transcription": "我每天早上都要喝杯茶.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_013.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_013.wav",
        "output_transcription": "他的名片上有很多頭銜.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_014.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_014.wav",
        "output_transcription": "這個市場的東西很便宜.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_015.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_015.wav",
        "output_transcription": "我們喜歡看電視連續劇.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_016.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_016.wav",
        "output_transcription": "他們非常熟習中國歷史.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_017.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_017.wav",
        "output_transcription": "這個人看起來彬彬有禮.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_018.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_018.wav",
        "output_transcription": "他的臉上長了很多麻疹.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_019.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_019.wav",
        "output_transcription": "他今年七月要參加考試.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_020.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_020.wav",
        "output_transcription": "他特別留意看天氣預報.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_021.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_021.wav",
        "output_transcription": "我昨天沒能參加招待會.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_022.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_022.wav",
        "output_transcription": "我忘了把參考書帶給你.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_023.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_023.wav",
        "output_transcription": "讓我們約個時間見面吧.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_024.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_024.wav",
        "output_transcription": "我想和您討論那個計劃.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_025.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_025.wav",
        "output_transcription": "我有事要和你們經理談.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_026.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_026.wav",
        "output_transcription": "我要搭乘本週五的飛機.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_027.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_027.wav",
        "output_transcription": "我要預定三個人的座位.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_028.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_028.wav",
        "output_transcription": "把這張卡片填好交給我.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_029.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_029.wav",
        "output_transcription": "每個人需要付十塊台幣.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_030.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_030.wav",
        "output_transcription": "他為你的考試成績擔心.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_031.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_031.wav",
        "output_transcription": "大多數北方人愛吃水餃.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_032.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_032.wav",
        "output_transcription": "聖誕節前信箱塞滿賀卡.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_033.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_033.wav",
        "output_transcription": "這個房間裡的燈光很暗.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_034.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_034.wav",
        "output_transcription": "外面的氣溫是零下十度.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_035.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_035.wav",
        "output_transcription": "他穿了一件灰格子上衣.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_036.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_036.wav",
        "output_transcription": "他裝修房子花了三萬塊.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_037.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_037.wav",
        "output_transcription": "學音樂的人需要些天賦.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_038.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_038.wav",
        "output_transcription": "大家有事都愛找他商量.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_039.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_039.wav",
        "output_transcription": "一大早他就在外面掃地.",
    },
    {
        "input_audio": "examples/ldv/mix/boy1_papercup_LDV_040.wav",
        "output_audio": "examples/ldv/spk/boy1_papercup_clean_040.wav",
        "output_transcription": "你出門時別忘了帶鑰匙.",
    },
]

# Process each test audio file
for input_audio in test_audio_files:
    # Generate output filename based on test audio file
    # Tag: ldv_40shot to distinguish from optical_denoised experiments
    test_file_name = os.path.basename(input_audio).replace('.wav', '').replace('boy1_papercup_', '')
    output_audio_path = f"examples/audio_enhancement_{test_file_name}_ldv_40shot_result.wav"

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
