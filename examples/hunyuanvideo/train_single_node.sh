#!/bin/bash
# CUDA_VISIBLE_DEVICES=6
export WANDB_MODE="offline"
export MODEL_PATH="/storage/ysh/Code/ID_Consistency/Code/2_offen_codes/0_temp_hf/HunyuanVideo/ckpt_diffusers"
export DATASET_PATH="/storage/ysh/Code/ID_Consistency/Code/2_offen_codes/0_temp_hf/HunyuanVideo/Disney-VideoGeneration-Dataset"
export OUTPUT_PATH="cogvideox-lora-single-node"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file /storage/ysh/Code/ID_Consistency/Code/2_offen_codes/0_temp_hf/HunyuanVideo/accelerate_config_machine_single.yaml \
  train.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column /storage/ysh/Code/ID_Consistency/Code/2_offen_codes/0_temp_hf/HunyuanVideo/Disney-VideoGeneration-Dataset/prompt_1.txt \
  --video_column /storage/ysh/Code/ID_Consistency/Code/2_offen_codes/0_temp_hf/HunyuanVideo/Disney-VideoGeneration-Dataset/videos_1.txt \
  --validation_prompt "DISNEY A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 100 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 320 \
  --width 512 \
  --fps 15 \
  --max_num_frames 61 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb