###ddpm_unet.sh PSEUDO###
env_name=${1}
model_name=${2:-"ddpm-unet"}
batch_size=${3:-16}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Check if batch_size is provided , if not use default value of 16
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=16
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_unconditional.py \
  --model_config_name_or_path ${model_name} \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --dataloader_num_workers 1 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}" \
  --train_batch_size=${batch_size} \
  --num_epochs=1 \
  --gradient_accumulation_steps=2 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \