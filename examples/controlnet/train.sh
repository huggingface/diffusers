###controlnet.sh PSEUDO###
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
moreh-switch-model -M 1 
# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
python train_controlnet.py \
    --pretrained_model_name_or_path ${model_name}  \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-hed \
    --dataset_name fusing/fill50k \
    --output_dir ${output_dir} \
    --log_dir "${log_dir}" \
    --learning_rate=1e-5 \
    --max_train_steps 10 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=5