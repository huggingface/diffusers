export MODEL_NAME="nvidia/Cosmos-Predict2.5-2B"
export DATA_DIR="datasets/cosmos_nemo_assets"
#export DATASET_NAME=""
export TOKENIZERS_PARALLELISM=false

accelerate launch --mixed_precision="bf16" train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME --revision pre-trained \
  --dataset_name=$DATASET_NAME --train_data_dir=$DATA_DIR \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="outputs/mydiffusers" \
  --validation_prompt="" \
  --report_to=wandb \
  --allow_tf32 --gradient_checkpointing
