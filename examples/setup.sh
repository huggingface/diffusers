base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
current_dir=$(pwd)

task=$1
env_name=$task
echo "# ========================================================= #"
echo "create env for task ${task}.."

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
    cd ..
    pip install -e .
    cd ${current_dir}/${task}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    echo -e "\\n" | update-moreh --torch 1.10.0 --target 23.6.0 --force
fi
