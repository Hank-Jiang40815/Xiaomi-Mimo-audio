"""
針對 Input Local Transformer 的微調腳本
專門處理嚴重噪聲和訊號缺損

使用方式:
    python finetune_input_local.py \
        --model_path models/MiMo-Audio-7B-Base \
        --train_data examples/ldv \
        --output_dir outputs/input_local_finetune
"""

import os
import argparse
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

from src.mimo_audio.mimo_audio import MimoAudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_input_local_modules(model):
    """檢查 Input Local Transformer 的結構"""
    logger.info("\n" + "="*80)
    logger.info("Input Local Transformer 模組結構")
    logger.info("="*80)
    
    input_local_modules = []
    for name, module in model.named_modules():
        if "input_local_transformer" in name:
            input_local_modules.append((name, type(module).__name__))
            logger.info(f"  {name}: {type(module).__name__}")
    
    logger.info(f"\n總共 {len(input_local_modules)} 個模組")
    return input_local_modules


def create_input_local_lora_config(args):
    """
    建立專門針對 Input Local Transformer 的 LoRA 配置
    """
    
    # Input Local Transformer 的目標模組
    # 基於 Qwen2Model 結構
    target_modules = [
        # Attention 層
        "input_local_transformer.layers.*.self_attn.q_proj",
        "input_local_transformer.layers.*.self_attn.k_proj",
        "input_local_transformer.layers.*.self_attn.v_proj",
        "input_local_transformer.layers.*.self_attn.o_proj",
        # MLP 層
        "input_local_transformer.layers.*.mlp.gate_proj",
        "input_local_transformer.layers.*.mlp.up_proj",
        "input_local_transformer.layers.*.mlp.down_proj",
    ]
    
    # 如果要包含 embeddings
    if args.include_embeddings:
        target_modules.extend([
            "speech_embeddings.*",
            "speech_group_downcast",
        ])
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    logger.info("\n" + "="*80)
    logger.info("LoRA 配置")
    logger.info("="*80)
    logger.info(f"  Rank (r): {args.lora_r}")
    logger.info(f"  Alpha: {args.lora_alpha}")
    logger.info(f"  Dropout: {args.lora_dropout}")
    logger.info(f"  Target modules: {len(target_modules)} patterns")
    logger.info(f"  Include embeddings: {args.include_embeddings}")
    
    return lora_config


def freeze_non_input_local_modules(model):
    """
    凍結除了 Input Local Transformer 之外的所有模組
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 只訓練 input_local_transformer 相關的參數
        if "input_local_transformer" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    
    logger.info("\n" + "="*80)
    logger.info("參數統計")
    logger.info("="*80)
    logger.info(f"  總參數: {total_params:,}")
    logger.info(f"  可訓練參數: {trainable_params:,}")
    logger.info(f"  可訓練比例: {100 * trainable_params / total_params:.4f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Input Local Transformer for severe noise"
    )
    
    # 模型參數
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    # 資料參數
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    
    # 輸出參數
    parser.add_argument("--output_dir", type=str, default="outputs/input_local_finetune")
    
    # 訓練參數
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="較高的學習率，因為只訓練部分層")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    # LoRA 參數
    parser.add_argument("--lora_r", type=int, default=32,
                        help="較高的 rank，因為降噪任務複雜")
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--include_embeddings", action="store_true",
                        help="是否也微調 speech embeddings")
    
    # 微調策略
    parser.add_argument("--target_component", type=str, default="input_local",
                        choices=["input_local", "input_local_with_embeddings", "full_encoder"],
                        help="要微調的元件")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="凍結 decoder (local_transformer)")
    
    # 優化參數
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    
    # 其他
    parser.add_argument("--inspect_only", action="store_true",
                        help="只檢查模型結構，不訓練")
    parser.add_argument("--wandb_project", type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Input Local Transformer 微調 - 針對嚴重噪聲")
    logger.info("="*80)
    
    # 1. 載入模型
    logger.info("\n📦 Loading MiMo-Audio model...")
    mimo = MimoAudio(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )
    
    model = mimo.model
    
    # 2. 檢查結構
    logger.info("\n🔍 Inspecting model structure...")
    inspect_input_local_modules(model)
    
    if args.inspect_only:
        logger.info("\n✅ Inspection completed. Exiting.")
        return
    
    # 3. 設定 LoRA
    logger.info("\n🎯 Setting up LoRA for Input Local Transformer...")
    lora_config = create_input_local_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    # 4. 如果需要，額外凍結某些模組
    if args.freeze_decoder:
        logger.info("\n🔒 Freezing decoder (local_transformer)...")
        for name, param in model.named_parameters():
            if "local_transformer" in name and "input_local" not in name:
                param.requires_grad = False
    
    # 顯示可訓練參數
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n✅ Trainable: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.4f}%)")
    
    # 5. 準備資料集
    logger.info("\n📊 Loading datasets...")
    from finetune_lora import AudioEnhancementDataset, AudioEnhancementCollator
    
    train_dataset = AudioEnhancementDataset(
        data_dir=args.train_data,
        tokenizer=mimo.tokenizer,
        audio_tokenizer=mimo.mimo_audio_tokenizer,
    )
    
    val_dataset = None
    if args.val_data:
        val_dataset = AudioEnhancementDataset(
            data_dir=args.val_data,
            tokenizer=mimo.tokenizer,
            audio_tokenizer=mimo.mimo_audio_tokenizer,
        )
    
    data_collator = AudioEnhancementCollator(mimo.tokenizer)
    
    # 6. 訓練參數
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_steps=200 if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        fp16=args.fp16,
        report_to="wandb" if args.wandb_project else "none",
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # 7. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 8. 開始訓練
    logger.info("\n🚀 Starting training...")
    logger.info(f"  Target: Input Local Transformer ({args.target_component})")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    train_result = trainer.train()
    
    # 9. 保存
    logger.info("\n💾 Saving model...")
    trainer.save_model()
    
    logger.info("\n✅ Training completed!")
    logger.info(f"📁 Model saved to: {args.output_dir}")
    
    # 10. 顯示訓練統計
    metrics = train_result.metrics
    logger.info("\n📊 Training metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
