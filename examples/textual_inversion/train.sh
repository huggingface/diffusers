export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/scratch/mp5847/diffusers_generated_datasets/van_gogh_50_attention_lr=1e-5_sd_v1.4/train"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<art-style>" --initializer_token="art" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/scratch/mp5847/diffusers_ckpt/textual_inversion/van_gogh_50_attention_lr=5.0e-04_sd_v1.4"