export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="../../../models/imagic"
export INPUT_IMAGE="imgs/Official_portrait_of_Barack_Obama.jpg"

accelerate launch train_imagic.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --input_image=$INPUT_IMAGE \
  --target_text="A photo of Barack Obama smiling with a big grin." \
  --seed=3434554 \
  --resolution=512 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --emb_learning_rate=1e-3 \
  --learning_rate=1e-6 \
  --emb_train_steps=500 \
  --max_train_steps=1000
