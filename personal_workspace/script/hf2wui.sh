###
 # @Author: Juncfang
 # @Date: 2023-02-24 10:30:06
 # @LastEditTime: 2023-03-02 13:41:40
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/script/hf2wui.sh
 #  
### 

export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export MODEL_PATH="/home/junkai/code/diffusers_fork/personal_workspace/dreambooth/experiments/2023-04-23T19:44:35-idphoto0410-6add-r1.3-cls_r-500-inpainting1-yanjie-new-ddim/models"
export CKPT_DIR="/home/junkai/code/stable-diffusion-webui/models/Stable-diffusion"
export CKPT_NAME="23-04-11-yanjie-new-ddim-500.safetensors"
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