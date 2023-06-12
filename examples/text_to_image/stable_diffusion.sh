###controlnet.sh PSEUDO###

base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
current_dir=$(pwd)
bs_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
env_name="stable-diffusion"


conda clean --all --force-pkgs-dir -y
conda create --name ${env_name} python=3.8 -y
source ${base_env}/etc/profile.d/conda.sh
conda activate ${env_name}


if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ]; then
    echo "installing requirements.."
    cd ../../
    pip install -e .
    cd ${bs_dir}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    update-moreh --torch 1.10.0 --target 23.5.0
else
    echo "The command is NOT running in the correct Conda environment."
    echo "Stop the training process."
    exit 1
fi
 
echo "training ${env_name}.."
python train_text_to_image.py \
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir ./${env_name}_outputs/ \

echo "training done. deleting env.."
conda deactivate
conda env remove -n ${env_name}
cd ${current_dir}