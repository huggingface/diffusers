###stable_diffusion.sh PSEUDO###
env_name=${1}
model_name=${2}
batch_size=${3}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Activate conda env
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
source ${base_env}/etc/profile.d/conda.sh
conda activate ${env_name}
moreh-switch-model -M 2

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
python train_text_to_image.py \
  --pretrained_model_name_or_path ${model_name} \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}" \