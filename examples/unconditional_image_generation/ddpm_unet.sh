###ddpm_unet.sh PSEUDO###

base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
current_dir=$(pwd)
env_name="ddpm_unet"


conda clean --all --force-pkgs-dir -y
conda create --name ${env_name} python=3.8 -y
source ${base_env}/etc/profile.d/conda.sh
conda activate ${env_name}


if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ]; then
    echo "installing requirements.."
    cd ../../
    pip install -e .
    cd ${current_dir}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    update-moreh --torch 1.10.0 --target 23.5.0
else
    echo "The command is NOT running in the correct Conda environment."
    echo "Stop the training process."
    exit 1
fi

 
echo "training ${env_name}.."
python train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --dataloader_num_workers 1 \
  --output_dir ./ddpm_unet_outputs/ \
  --train_batch_size=64 \
  --num_epochs=1 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \

echo "training done. deleting env.."
conda deactivate
conda env remove -n ${env_name}