Training command:

```bash
accelerate launch train_diffusion_orpo_sdxl_lora.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-orpo" \
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

Based on https://github.com/huggingface/trl/pull/1435/. 

Bigger training command:

```bash
accelerate launch --multi_gpu train_diffusion_orpo_sdxl_lora_wds.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --dataset_path="pipe:aws s3 cp s3://diffusion-preference-opt/{00000..00644}.tar -" \
  --output_dir="diffusion-sdxl-orpo-wds" \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --dataloader_num_workers=8 \
  --learning_rate=3e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=2000 \
  --run_validation --validation_steps=500 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```