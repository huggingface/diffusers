###ddpm_unet.sh PSEUDO###
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
bs_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
current_dir=$(pwd)

task=${1:-"unconditional_image_generation"}
env_name=$task
model_name=${2:-"ddpm-unet"}
batch_size=${3:-16}
output_dir=${4:-"outputs"}
log_dir=${5:-"${current_dir}/logs/${env_name}/${model_name}.json"}
run_mode=${6:-2}
echo "# ========================================================= #"
echo "create env for model ${model_name}.."

if conda env list | grep -q -E "^$env_name\s"; then
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
else
    conda clean --all --force-pkgs-dir -y
    conda create --name ${env_name} python=3.8 -y
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
    install_requirements=1
fi
echo "environment name: ${env_name}"

if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ] && [ "$install_requirements" == "1" ]; then
    echo "installing requirements in conda env ${env_name}.."
    cd ../../
    pip install -e .
    cd ${bs_dir}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    update-moreh --torch 1.10.0 --target 23.6.0 --force
fi

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

if [ "$run_mode" = 1 ]; then
    echo "# ========================================================= #"
    echo "training ${model_name} done. deleting env.."
    conda deactivate
    conda env remove -n ${env_name}
    cd ${current_dir}
fi