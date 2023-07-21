export MODEL_DIR="stabilityai/stable-diffusion-xl-base-0.9"
export OUTPUT_DIR="controlnet-0-9-canny"

# --max_train_steps=15000 \
accelerate launch train_controlnet_webdatasets.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
 --output_dir=$OUTPUT_DIR \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=30000 \
 --max_train_samples=12000000 \
 --dataloader_num_workers=4 \
 --validation_image "./c_image_0.png" "./c_image_1.png" "./c_image_2.png" "./c_image_3.png" "./c_image_4.png" "./c_image_5.png" "./c_image_6.png" "./c_image_7.png" \
 --validation_prompt "beautiful room" "two paradise birds" "a snowy house behind a forest" "a couple watching a romantic sunset" "boats in the Amazonas" "a beautiful face of a woman" "a skater in Brooklyn" "a tornado in Iowa" \
 --train_shards_path_or_url "pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-data/{00000..01208}.tar -" \
 --eval_shards_path_or_url "pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-data/{01209..01210}.tar -" \
 --proportion_empty_prompts 0.5 \
 --validation_steps=1000 \
 --train_batch_size=12 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --gradient_accumulation_steps=1 \
 --seed=42 \
 --report_to="wandb" \
 --push_to_hub
