export MODEL_NAME="../../../mus2vid/models/stable-diffusion-v1-5"
export INSTANCE_DIR="portraits"
export CLASS_DIR="portrait_photo_class_imgs"
export OUTPUT_DIR="sd-portraits-v1-5_v2"


#  --train_text_encoder \

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a sks portrait photo" \
  --class_prompt="a portrait photo" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=1000 \
  --max_train_steps=1200
