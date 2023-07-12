###stable_diffusion.sh PSEUDO###
output_dir=outputs
current_dir=$(pwd)

while getopts t:m:b:o:l: flag
do
    case "${flag}" in
        t) env_name=${OPTARG};;
        m) model_name=${OPTARG};;
        b) batch_size=${OPTARG};;
        o) output_dir=${OPTARG};;
        l) log_dir=${OPTARG};;
    esac
done

# Check if log_dir is provided , if not use default value of current_dir/logs/${model_name}.json
if [ -z "$log_dir" ]
then
    log_dir=${current_dir}/logs/${model_name}.json
    mkdir -p ${current_dir}/logs
fi

# Check if $batch_size is provided , if not use default value of 1
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=6
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_text_to_image_mlflow.py \
  --pretrained_model_name_or_path ${model_name} \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size  ${batch_size}\
  --gradient_accumulation_steps=4 \
  --max_train_steps=30 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}" \
  --logging_steps=100 \