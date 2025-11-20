# Outputs Summary (not tracked)

- `outputs/optical_lora_v2_normbeforemel_r32_e100_distill_e30_lci005/`
  - best val loss ~1.70 @ epoch 19 (train ~1.79, val_code_idx ~2e-12)
  - checkpoints: best_model.pt, checkpoint_epoch_10/20/30.pt
  - training_history.json present (no training.log tee)
  - inference sample: outputs/test_inference_normbeforemel_v2_idx_distill_e30_lci005/enhanced_050.wav

- `outputs/optical_lora_v2_normbeforemel_r32_e100_v2_idx_e30_lci005/`
  - best val loss ~0.91 @ epoch 16 (train ~2.10, val_code_idx ~10.9)
  - checkpoints: best_model.pt, checkpoint_epoch_10/20/30.pt
  - training_history.json present (no training.log tee)
  - inference sample: outputs/test_inference_normbeforemel_v2_idx_e30_lci005/enhanced_050.wav

Note: outputs are git-ignored; this file only records where they are and key metrics.
