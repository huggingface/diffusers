base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (writable)//' | tr -d ' ')
current_dir=$(pwd)

task=$1
env_name=$task


if conda env list | grep -q -E "^$env_name\s"; then
    echo "# ========================================================= #"
    echo "Env for task ${task} exists, activate it.."
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
    
else
    echo "# ========================================================= #"
    echo "Create env for task ${task}.."
    conda clean --all --force-pkgs-dir -y
    conda create --name ${env_name} python=3.8 -y
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
    install_requirements=1
    
fi

if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ] && [ "$install_requirements" == "1" ]; then
    echo "installing requirements in conda env ${env_name}.."
    cd ..
    pip install -e .
    cd ${current_dir}/${task}
    pip install -r requirements.txt
fi
