###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-02-20 10:11:21
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="0"
export EXPERIMENT_NAME="idphoto0216_4seg512"
# export PROMPT="Passport photo with correct ID or passport size, profile picture with correct ID or passport size photo, a close up of a person wearing glasses, jewish young man with glasses, slight nerdy smile, cute slightly nerdy smile, smiling and looking directly, halfbody headshot, headshot photo, miles johnstone, high quality portrait, very slightly smiling, headshot portrait, slightly smiling, happily smiling at the camera, smiling slightly, large eyes and menacing smile."
# export PROMPT="an asian woman with long hair and a blue shirt"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, a photo of a woman with symmetrical eyes, solid white background, canon 5d"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman with long hair, solid white background, canon 5d"
export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman, solid white background, canon 5d"
# export PROMPT="a photo of a man"
# export PROMPT="a photo of a person"
export BASE_SEED=-1
export IMAGE_NUM=260
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=60
export GUIDANCE_SCALE=7
export NEGATIVE_PROMPT="Asymetrical eyes, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

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

CUDA_VISIBLE_DEVICES="$GPU_ID" python $PROJECT_DIR/personal_workspace/inference.py \
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