your_local_path='output'

hf download Efficient-Large-Model/SANA_Sprint_1.6B_1024px_teacher_diffusers  --local-dir $your_local_path/SANA_Sprint_1.6B_1024px_teacher_diffusers

# or Sana_Sprint_0.6B_1024px_teacher_diffusers

python train_sana_sprint_diffusers.py \
    --pretrained_model_name_or_path=$your_local_path/SANA_Sprint_1.6B_1024px_teacher_diffusers \
    --output_dir=$your_local_path \
    --mixed_precision=bf16 \
    --resolution=1024 \
    --learning_rate=1e-6 \
    --max_train_steps=30000 \
    --dataloader_num_workers=8 \
    --dataset_name='brivangl/midjourney-v6-llava' \
    --file_path data/train_000.parquet data/train_001.parquet data/train_002.parquet \
    --checkpointing_steps=500 --checkpoints_total_limit=10 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --seed=453645634 \
    --train_largest_timestep \
    --misaligned_pairs_D \
    --gradient_checkpointing \
    --resume_from_checkpoint="latest" \


