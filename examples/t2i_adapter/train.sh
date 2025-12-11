#!/bin/bash

MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"

# ----- Hyperparameters -----
RES=512
LR=1e-7
BS=4
GAS=4
TEMP=1.0
SEED=42
STEP=10000

# ---- Auto-generate run name ----
DATE=$(date +"%m%d_%H%M")
RUN_NAME="sdxl_GMA_withFiLM_t${TEMP}_res${RES}_lr${LR}_bs${BS}x${GAS}_seed${SEED}_step${STEP}"

OUTPUT_DIR="/home/ubuntu/gate-your-sketch-training_output/${RUN_NAME}"

echo ">>> OUTPUT_DIR = $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# wandb login # 1ea8d30af7d9fdc65677729285a0773003477654
accelerate launch train_t2i_adapter_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="hengyiqun/10623-project-coco17" \
 --GMA_temperature=$TEMP \
 --sketch_adapter_path="TencentARC/t2i-adapter-sketch-sdxl-1.0" \
 --depth_adapter_path="TencentARC/t2i-adapter-depth-midas-sdxl-1.0" \
 --image_column="image" \
 --sketch_image_column="sketch" \
 --depth_image_column="depth" \
 --resolution=$RES \
 --mixed_precision="fp16" \
 --learning_rate=$LR \
 --max_train_steps=$STEP \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100000 \
 --train_batch_size=$BS \
 --gradient_accumulation_steps=$GAS \
 --report_to="wandb" \
 --seed=$SEED \
 --checkpointing_steps=1000 \
 --resume_from_checkpoint=latest

 #  --mixed_precision="fp16" \