###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-04-11 10:06:36
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="3"
export EXPERIMENT_NAME="idphoto0507_cutoff_6add_r1.3_wb"
# export PROMPT="Passport photo with correct ID or passport size, profile picture with correct ID or passport size photo, a close up of a person wearing glasses, jewish young man with glasses, slight nerdy smile, cute slightly nerdy smile, smiling and looking directly, halfbody headshot, headshot photo, miles johnstone, high quality portrait, very slightly smiling, headshot portrait, slightly smiling, happily smiling at the camera, smiling slightly, large eyes and menacing smile."
# export PROMPT="an asian woman with long hair and a blue shirt"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, a photo of a woman with symmetrical eyes, solid white background, canon 5d"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman with long hair, solid white background, canon 5d"
# experiment
# export PROMPT="(extremely detailed CG unity 8k wallpaper), (masterpiece), (best quality), (ultra-detailed), (best illustration), (best shadow), ultra-high res, close-up,  (photorealistic:1.4), 1girl at home, ((very oversize shirt, buttoned shirt, open shirt)), (man shirt), no bra, collarbone, no panties,  (small breasts:1.2), small nipples,  (light silver hair:1.2), looking at viewer, light smile, upper body, makeup, <lora:koreanDollLikeness_v10:0.3>,"
# export PROMPT="art by xyzjz, 8k, HD, photorealistic, a photo of an Asian man with solid white background"
# GOOD 
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian woman, solid white background, canon 5d"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian man, white background, canon 5d"
# export PROMPT="Symmetry!! highly detailed, 8k, HD, photorealistic, an asian man with black suit red tie and white shirt, white background, canon 5d"
# export PROMPT="a photo of an Asian woman in a black suit and white shirt and red tie with medium length hair without glasses and smile, white background, art by xyzjz"
export PROMPT="a photo of an Asian woman in a red suit, and blue shirt, and red tie, with medium length hair without glasses and smile,  with white background, art by xyzjz"
# export PROMPT="a photo of a man"
# export PROMPT="a photo of a person"
export INFER_FILE_NAME="inference_cutoff" # ["inference", "inference_cutoff"]
export BASE_SEED=-1
export IMAGE_NUM=20
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=60
export GUIDANCE_SCALE=7
export SAMPLER_METHOD="DF" #["DDIM", "DDPM", "UniPC", "PNDM", "DF"]
# export NEGATIVE_PROMPT="Asymetrical eyes, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
export NEGATIVE_PROMPT="rich background, colorful background, Asymetrical eyes, bad anatomy, bad hands, error, missing fingers, cropped, worst quality, low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
# export NEGATIVE_PROMPT="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low res, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, bad legs, error legs, bad feet, malformed limbs, extra limbs"

export MODEL_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"
# export MODEL_DIR="/home/junkai/code/diffusers_fork/personal_workspace/base_model/chilloutmix"
export OUTPUT_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/outputs"
if [[ ! -d $MODEL_DIR ]]; then
    MODEL_DIR=$EXPERIMENT_NAME
fi
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
    echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python $PROJECT_DIR/personal_workspace/$INFER_FILE_NAME.py \
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
--sampler_method "$SAMPLER_METHOD"\