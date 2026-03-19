export MODEL_NAME="nvidia/Cosmos-Predict2.5-2B"
export DATA_DIR="datasets/benchmark_train/gr1"
revision='post-trained'
echo $revision

export TOKENIZERS_PARALLELISM=false
accelerate launch --mixed_precision="bf16" train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME --revision $revision \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=1 \
  --num_train_epochs=500 --checkpointing_epochs=100 \
  --seed=0 \
  --output_dir="outputs/gr1" \
  --report_to=wandb \
  --height 432 --width 768 \
  --allow_tf32 --gradient_checkpointing \
  --lora_rank 32 --lora_alpha 32 \
