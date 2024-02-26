export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"


CUDA_VISIBLE_DEVICES="1" accelerate launch --num_processes 1 --main_process_port 29502  --mixed_precision="fp16"  train_inpainting.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 --seed=42 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-inpaint-2" --validation_size=3 --validation_epochs=1 --report_to="wandb"