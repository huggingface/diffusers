###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-02-28 10:24:14
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="3"
export EXPERIMENT_NAME="idphoto0216_5manual_r1.3"
# export PROMPT="Passport photo with correct ID or passport size, profile picture with correct ID or passport size photo, a close up of a person wearing glasses, jewish young man with glasses, slight nerdy smile, cute slightly nerdy smile, smiling and looking directly, halfbody headshot, headshot photo, miles johnstone, high quality portrait, very slightly smiling, headshot portrait, slightly smiling, happily smiling at the camera, smiling slightly, large eyes and menacing smile."
# export PROMPT="an asian woman with long hair and a blue shirt"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, a photo of a woman with symmetrical eyes, solid white background, canon 5d"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman with long hair, solid white background, canon 5d"
# experiment
# export PROMPT="RAW photo, a close up portrait photo of woman soft lighting, high quality, film grain, Fujifilm XT3"
export PROMPT="art by xyzjz, 8k, HD, photorealistic, a photo of an Asian man with solid white background"
# GOOD 
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman, solid white background, canon 5d"
# export PROMPT="a photo of a man"
# export PROMPT="a photo of a person"
export BASE_SEED=-1
export IMAGE_NUM=1260
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=60
export GUIDANCE_SCALE=7
export NEGATIVE_PROMPT="Asymetrical eyes, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
# export NEGATIVE_PROMPT="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

export MODEL_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"
# export MODEL_DIR="/home/junkai/code/diffusers_fork/personal_workspace/base_model/Realistic_Vision_V1.3"
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