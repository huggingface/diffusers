###instruct_pix2pix.sh PSEUDO###
env_name=${1}
model_name=${2}
batch_size=${3}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}
DATASET_ID="fusing/instructpix2pix-1000-samples"

# Check if batch_size is provided , if not use default value of 4
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=4
fi

moreh-switch-model -M 1

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_instruct_pix2pix.py \
    --pretrained_model_name_or_path ${model_name} \
    --dataset_name=$DATASET_ID \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --max_train_steps=100 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir ${output_dir} \
    --log_dir "${log_dir}" \