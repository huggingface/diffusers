export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="/scratch/mp5847/diffusers_generated_datasets/van_gogh_bedroom"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="$DATASET_NAME" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2000 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/scratch/mp5847/diffusers_ckpt/van_gogh_bedroom_lora_lr=1e-04_1_img"