###controlnet.sh PSEUDO###
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

moreh-switch-model -M 1

# Check if $batch_size is provided , if not use default value of 5
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=8
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_controlnet.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5  \
    --controlnet_model_name_or_path ${model_name}  \
    --dataset_name fusing/fill50k \
    --output_dir ${output_dir} \
    --log_dir "${log_dir}" \
    --learning_rate=1e-5 \
    --max_train_steps 10 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size ${batch_size} \