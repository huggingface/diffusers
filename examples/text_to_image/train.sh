export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/scratch/mp5847/diffusers_generated_datasets/van_gogh_5000_sd_v1.4"

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="$DATASET_NAME" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=1000 \
  --output_dir="/scratch/mp5847/diffusers_ckpt/van_gogh_5000_attention_lr=1e-5_sd_v1.4" 