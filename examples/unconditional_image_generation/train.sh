###ddpm_unet.sh PSEUDO###
env_name=${1}
model_name=${2:-"ddpm-unet"}
batch_size=${3:-16}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Activate conda env
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
source ${base_env}/etc/profile.d/conda.sh
conda activate ${env_name}

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
python train_unconditional.py \
  --model_config_name_or_path ${model_name} \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --dataloader_num_workers 1 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}/${model_name}.json" \
  --train_batch_size=${batch_size} \
  --num_epochs=1 \
  --gradient_accumulation_steps=2 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \