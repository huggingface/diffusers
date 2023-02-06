###
 # @Author: Juncfang
 # @Date: 2023-02-03 15:45:03
 # @LastEditTime: 2023-02-06 10:12:12
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/finetune/finetune_t2i_localdata.sh
 #  
### 
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export CUDA_VISIBLE_DEVICES="1"
export EXPERIMENT_NAME="idphoto-test"
export TRAIN_DIR="/RAID5/user/junkai/data/IDPhoto/IDphoto-blip2-captions"
export OUTPUT_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"

if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
    echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

cd ../examples/text_to_image && \
accelerate launch train_text_to_image.py \
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
  --output_dir=$OUTPUT_DIR