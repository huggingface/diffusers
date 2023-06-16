###dreambooth.sh PSEUDO###
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

# Check if data is downloaded
data_dir=/nas/common_data/huggingface/dreambooth/dog
echo "# ========================================================= #"
if [ -d "$data_dir" ] && [ "$(ls -A $data_dir)" ]; then
  echo "data is already in $data_dir"
else
  echo "downloading data.."
  python /nas/thuchk/repos/diffusers/examples/dreambooth/download_data.py
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_dreambooth.py \
  --pretrained_model_name_or_path ${model_name} \
  --instance_data_dir $data_dir \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size ${batch_size} \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --output_dir ${output_dir}  \
  --log_dir "${log_dir}" \