export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/ubuntu/diffusion_tests/data/alvan"
export CLASS_DIR="/home/ubuntu/diffusion_tests/data/dog"
export OUTPUT_DIR="/home/ubuntu/diffusion_tests/models/dog2"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME --use_auth_token \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="A sks dog" \
  --class_prompt="A dog" \
  --seed=3434554 \
  --resolution=512 \
  --center_crop \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=12 \
  --sample_batch_size=4 \
  --max_train_steps=800
