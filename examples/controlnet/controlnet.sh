###controlnet.sh PSEUDO###

base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
current_dir=$(pwd)
env_name="controlnet"


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
python train_controlnet.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5  \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-hed \
    --dataset_name fusing/fill50k \
    --output_dir ./${env_name}_outputs/ \
    --learning_rate=1e-5 \
    --max_train_steps 10 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=5

echo "training done. deleting env.."
conda deactivate
conda env remove -n ${env_name}