###
 # @Author: Juncfang
 # @Date: 2022-12-30 09:57:16
 # @LastEditTime: 2023-02-28 16:32:49
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /diffusers_fork/personal_workspace/dreambooth/train_dreambooth.sh
 #  
### 
printf -v DATE '%(%Y-%m-%dT%H:%M:%S)T' -1
export CURDIR="$( cd "$( dirname $0 )" && pwd )"
export PROJECT_DIR="$( cd "$CURDIR/../.." && pwd )"

export GPU_ID="1"
export EXPERIMENT_NAME="idphoto0216-5manual-r1.3-man_r-1200-seg-align3-u9-white"
export EXPERIMENT_NAME="$DATE-$EXPERIMENT_NAME"
export INSTANCE_DIR="$CURDIR/datasets/u9_align3_seg"
export MAX_STEP=1200
export CLASS_NAME="man" # "man", "man2", "<ID-PHOTO>", "woman", "person", "cat", "dog" ...
# MODEL_NAME ect. "CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/home/junkai/code/diffusers_fork/personal_workspace/finetune/experiments/idphoto0216_5manual_r1.3/models" 

export CLASS_DIR="$CURDIR/class_data/$CLASS_NAME"
export OUTPUT_DIR="$CURDIR/experiments/$EXPERIMENT_NAME/models"
if [[ ! -d $INSTANCE_DIR ]]; then
    echo "Can not found required INSTANCE_DIR at ' $INSTANCE_DIR '!"
    exit 1
fi
if [[ ! -d $CLASS_DIR ]]; then
    echo "Can not found required CLASS_DIR at '$CLASS_DIR' !"
    exit 1
fi
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
elif [[ ! -d $OUTPUT_DIR ]]; then
    echo "$OUTPUT_DIR already exists but is not a directory" 1>&2
fi

# export INSTANCE_PROMPT="<?>"
export INSTANCE_PROMPT="a photo of a <?> $CLASS_NAME"
export CLASS_PROMPT="a photo of a $CLASS_NAME"

# print some information
echo \
"
================================================================
EXPERIMENT_NAME: $EXPERIMENT_NAME
INSTANCE_DIR: $INSTANCE_DIR 
MAX_STEP: $MAX_STEP 
MODEL_NAME: $MODEL_NAME 
INSTANCE_PROMPT: $INSTANCE_PROMPT 
CLASS_PROMPT: $CLASS_PROMPT 
CLASS_DIR: $CLASS_DIR
================================================================
"

# train
CUDA_VISIBLE_DIVICES=$GPU_ID \
accelerate launch $PROJECT_DIR/examples/dreambooth/train_dreambooth.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--class_data_dir=$CLASS_DIR \
--prior_loss_weight=0.5 \
--instance_prompt="$INSTANCE_PROMPT" \
--class_prompt="$CLASS_PROMPT" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--use_8bit_adam \
--learning_rate=2e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_class_images=260 \
--max_train_steps=$MAX_STEP \
--mixed_precision="fp16" \
--gradient_checkpointing \
--logging_dir="../logs" \
--train_text_encoder \
--with_prior_preservation \

# --enable_xformers \
# --enable_xformers_memory_efficient_attention