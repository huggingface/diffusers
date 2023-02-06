###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-02-06 10:16:28
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"

export GPU_ID="1"
export EXPERIMENT_NAME="idphoto-test"
# export PROMPT="Passport photo with correct ID or passport size, profile picture with correct ID or passport size photo, a close up of a person wearing glasses, jewish young man with glasses, slight nerdy smile, cute slightly nerdy smile, smiling and looking directly, halfbody headshot, headshot photo, miles johnstone, high quality portrait, very slightly smiling, headshot portrait, slightly smiling, happily smiling at the camera, smiling slightly, large eyes and menacing smile."
export PROMPT="a photo of a person"
export BASE_SEED=0
export IMAGE_NUM=10
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7
export NEGATIVE_PROMPT="nsfw,lowers,bad anatomy, bad hands, text,error,missing fingers,extra digit,\
    fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry"

export MODEL_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"
export OUTPUT_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/outputs"
if [[ ! -d $MODEL_DIR ]]; then
    MODEL_DIR=$EXPERIMENT_NAME
fi
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
    echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python inference.py \
--pretrained_model_name_or_path $MODEL_DIR \
--prompt "$PROMPT" \
--negative_prompt "$NEGATIVE_PROMPT" \
--base_seed $BASE_SEED \
--image_num $IMAGE_NUM \
--width $IMAGE_WIDTH \
--height $IMAGE_HEIGHT \
--num_inference_steps $NUM_INFERENCE_STEPS \
--guidance_scale $GUIDANCE_SCALE \
--output_dir $OUTPUT_DIR \