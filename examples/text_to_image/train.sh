###stable_diffusion.sh PSEUDO###
env_name=${1}
model_name=${2}
batch_size=${3}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Check if batch_size is provided , if not use default value of 1
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=1
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_text_to_image.py \
  --pretrained_model_name_or_path ${model_name} \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_siz \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}" \