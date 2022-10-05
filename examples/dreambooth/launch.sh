export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/ubuntu/diffusion_tests/data/shivam"
export CLASS_DIR="/home/ubuntu/diffusion_tests/data/person"
export OUTPUT_DIR="/home/ubuntu/diffusion_tests/models/shivam"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="sks person" \
  --class_prompt="person" \
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
  --num_class_images=20 \
  --sample_batch_size=4 \
  --max_train_steps=1000
