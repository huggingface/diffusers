###
 # @Author: Juncfang
 # @Date: 2023-02-24 10:30:06
 # @LastEditTime: 2023-02-27 19:43:41
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/script/hf2wui.sh
 #  
### 

export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export MODEL_PATH="/home/junkai/code/diffusers_fork/personal_workspace/dreambooth/experiments/2023-02-23T10:36:55-idphoto0216-5manual-1200-seg-align3-u9-white/models"
export CKPT_DIR="/home/junkai/code/stable-diffusion-webui/models/Stable-diffusion"
export CKPT_NAME="u91200good.safetensors"
if [[ ! -d $CKPT_DIR ]]; then
    mkdir -p $CKPT_DIR
fi

export CKPT_PATH="$CKPT_DIR/$CKPT_NAME"
echo $CKPT_PATH
python $PROJECT_DIR/scripts/convert_diffusers_to_original_stable_diffusion.py \
--model_path "$MODEL_PATH" \
--checkpoint_path "$CKPT_PATH" \
--use_safetensors \
# --half \