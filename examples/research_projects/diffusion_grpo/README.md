# Diffusion Model Alignment Using GRPO


This directory provides LoRA implementations of Diffusion [GRPO](https://arxiv.org/abs/2402.03300) an RL based alignment method which is a variant of Proximal Policy Optimization (PPO) in the diffusion model setting.

## SDXL training command

```bash
accelerate launch train_diffusion_grpo_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-dpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --run_validation --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```
