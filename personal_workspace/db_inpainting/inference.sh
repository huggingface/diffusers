###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-03-01 17:22:01
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/db_inpainting/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="1"
export EXPERIMENT_NAME="2023-03-01T17:00:50-test-mask"
# Test
export PROMPT="a photo of <?> woman."

export IMG_PATH="./test_data/2i.png"
export MSK_PATH="./test_data/2m.png"
export BASE_SEED=-1
export IMAGE_NUM=10
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7
export NEGATIVE_PROMPT="rich background, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy rich background, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy rich background, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

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

CUDA_VISIBLE_DEVICES="$GPU_ID" python ./inference.py \
--image_path "$IMG_PATH" \
--mask_path "$MSK_PATH" \
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