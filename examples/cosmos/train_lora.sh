export MODEL_NAME="nvidia/Cosmos-Predict2.5-2B"
export DATA_DIR="datasets/benchmark_train/gr1"
export TOKENIZERS_PARALLELISM=false
revision='post-trained'
echo $revision

accelerate launch --mixed_precision="bf16" train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME --revision $revision \
  --dataset_name=$DATASET_NAME --train_data_dir=$DATA_DIR \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_epochs=10 \
  --seed=0 \
  --output_dir="outputs/gr1_ep100" \
  --validation_prompt="" \
  --report_to=wandb \
  --height 432 --width 768 \
  --allow_tf32 --gradient_checkpointing \
