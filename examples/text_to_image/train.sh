export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="/scratch/mp5847/diffusers_generated_datasets/elon_musk"

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
  --checkpointing_steps=500 \
  --output_dir="/scratch/mp5847/diffusers_ckpt/elon_musk_full_lr=1e-05" 