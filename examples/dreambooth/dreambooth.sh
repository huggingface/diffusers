#!/bin/sh

MODEL_NAME=/store/sd/diffusers_models/stable-diffusion-2-1
DATA_DIR=/store/sd/training/drbolick_768/filter_1
OUT_DIR=/store/sd/training/out/test
ACCELERATE_LOG_LEVEL="INFO"
LOG_LEVEL="INFO"
PYTHONUNBUFFERED=1

python train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --instance_prompt="drbolick style" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --checkpointing_steps=10 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --samples_per_checkpoint=4 \
  --sample_steps=40 \
  --sample_prompt="a woman sitting among trees against the horizon. drbolick style" \
  --sample_seed=980273

#   --gradient_checkpointing
  # --class_prompt="art style" \
  # --with_prior_preservation