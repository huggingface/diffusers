###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-04-11 18:26:40
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/dreambooth/inference.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="2"
export EXPERIMENT_NAME="2023-04-23T19:44:35-idphoto0410-6add-r1.3-cls_r-500-inpainting1-yanjie-new-ddim"
# Test
export PROMPT="Symmetry!!,highly detailed, digital photo, a photo of a <?> woman in a suit, solid white background"
# export PROMPT="Symmetry!!,highly detailed, digital photo, good illumination,  photo of a <?> man with short hair in a suit, solid white background"
# export PROMPT="Symmetry!!,highly detailed digital photo, a photo of a <?> woman in a suit and a shirt, solid blue background,portrait,masterpiece,sharp"
# export PROMPT=" <?> man, Symmetry!!, highly detailed, digital photo, a photo of a <?> man in a suit and tie, solid red background, smile"
# export PROMPT=" <?> woman, Symmetry!!, highly detailed, digital photo, a photo of a <?> woman in a suit and tie, solid red background, smile"
# export PROMPT=" a photo of <?> woman art by xyzjz, highly detailed, digital photo, a photo of a <?> woman in a suit and tie, solid red background, smile"
# export PROMPT="a photo of <?> woman"
# export PROMPT="a asian with short hair wearing a black t - shirt"
# GOOD
# export PROMPT=" Symmetry!!, highly detailed, digital photo, a photo of a <?> woman in a suit and tie, solid red background"
# OTHER
# export PROMPT="solid color background. <ID-PHOTO>!! a photo of a <?> man with solid color background with <ID-PHOTO>, red background, <ID-PHOTO>"
# export PROMPT="a photo of a <?> man"
# export PROMPT="a photo of a <?> woman"
# export PROMPT="Symmetry!!,highly detailed, digital photo, a photo of a <?> man with long hair, solid white backgroundSymmetry!!,highly detailed, digital photo, a photo of a <?> man with long hair, solid white backgroundSymmetry!!,highly detailed, digital photo, a photo of a <?> man with long hair, solid white backgroundSymmetry!!,highly detailed, digital photo, a photo of a <?> man with long hair, solid white background"
# export PROMPT="a photo of a <?> man"
# export PROMPT="Solid white background. Symmetry!!,highly detailed, digital photo, a photo of a <?> man in a suit and tie, solid white background"
# export PROMPT="Symmetry!!,highly detailed, digital photo, a photo of a <?> woman in a suit and tie, solid white background"
# export PROMPT="Symmetry!!,highly detailed, digital photo, a photo of a <?> man in a suit and tie, solid red background"
# export PROMPT="A photo of a <?> man with long hair, in a suit and tie. Symmetry!!,highly detailed, solid red background."
# export PROMPT="A photo of a <?> man with smile, in a suit and tie. Symmetry!!,highly detailed, solid red background."
# export PROMPT="A photo of a <?> man without glasses, in a suit and tie. Symmetry!!,highly detailed, solid red background."
# export PROMPT="A photo of a <?> man in a suit and tie with red background. Symmetry!!,highly detailed"
# export PROMPT="A photo of a <?> man in a suit and tie with red background. looks exactly like <?> man!!, Symmetry!!,highly detailed"
# reference
# export PROMPT="Symmetry!!,highly detailed, a close-up of a <?> man looking forward with red background, high quality portrait, a young <?> man in a suit and tie with black hair, highly face detail, solid red background, best quality, very highly detailed!! highly detailed !!"
# export PROMPT="<?> man, Symmetry!!,highly detailed,  <?> photo, a photo of a <?> man in a black jacket posing for a portrait against a blue background, solid blue background"
# export PROMPT="Symmetry!!, <?> man highly detailed, <ID-PHOTO> a <?> man in a suit and tie against a blue background"
# export PROMPT="Symmetry!!, <?> man highly detailed, <ID-PHOTO> a <?> man in a suit and tie against a blue background"
# export PROMPT="<ID-PHOTO>!!!, <ID-PHOTO>, selfie portrait of <?>, pure black background"
export BASE_SEED=-1
export IMAGE_NUM=10
export IMAGE_WIDTH=512
export IMAGE_HEIGHT=512
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7
export SAMPLER_METHOD="DF" #["DDIM", "DDPM", "UniPC", "PNDM", "DF"]
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
--sampler_method "$SAMPLER_METHOD"\