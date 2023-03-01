###
 # @Author: Juncfang
 # @Date: 2023-02-24 10:30:06
 # @LastEditTime: 2023-02-27 15:51:13
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/script/wui2hf.sh
 #  
### 

export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export CKPT_PATH="/home/junkai/code/stable-diffusion-webui/models/Stable-diffusion/Realistic_Vision_V1.3.ckpt"
export HF_DIR="/home/junkai/code/diffusers_fork/personal_workspace/base_model/Realistic_Vision_V1.3"
if [[ ! -d $HF_DIR ]]; then
    mkdir -p $HF_DIR
fi

python $PROJECT_DIR/scripts/convert_original_stable_diffusion_to_diffusers.py \
--checkpoint_path "$CKPT_PATH" \
--dump_path "$HF_DIR" \
--to_safetensors \