###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-04-10 20:34:30
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/finetune_t2i_localdata.sh
 #  TODO: combine finetune_t2i_hubdata to this script
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="/home/junkai/code/diffusers_fork/personal_workspace/base_model/Realistic_Vision_V1.3"

export CUDA_VISIBLE_DEVICES="3"
export TRAIN_FILE="train_text_to_image_ddim_cutoff" #[train_text_to_image, train_text_to_image_ddim, train_text_to_image_ddim_cutoff]
# export EXPERIMENT_NAME="idphoto-<ID-PHOTO>"
export EXPERIMENT_NAME="idphoto0507_cutoff_6add_r1.3_wb"
# export TRAIN_DIR="/RAID5/user/junkai/data/IDPhoto/IDphoto-blip2-captions"
export TRAIN_DIR="/home/junkai/data/IDPhoto0507_6add_cutoff/IDphoto-blip2-captions"
# export TRAIN_DIR="/home/junkai/data/IDPhoto0302_6add/IDphoto-blip2-captions"
export OUTPUT_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"

if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
    echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

cd $PROJECT_DIR/examples/text_to_image && \
accelerate launch $TRAIN_FILE.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --logging_dir="../logs" \
#   --enable_xformers_memory_efficient_attention \