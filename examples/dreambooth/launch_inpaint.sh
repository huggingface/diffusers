export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export OUTPUT_DIR="../../../models/dress_inpainting"

accelerate launch train_inpainting_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_batch_size=2 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=300 \
  --sample_batch_size=4 \
  --max_train_steps=15000 \
  --save_interval=1000 \
  --save_min_steps=6000 \
  --save_infer_steps=35 \
  --concepts_list="concepts_list.json" \
  --not_cache_latents \
  --hflip
