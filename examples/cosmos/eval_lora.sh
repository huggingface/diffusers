export DATA_DIR="gr1_dataset/test"
export LORA_DIR=YOUR_ADAPTER_DIR
export OUT_DIR=YOUR_EVAL_OUTPUT_DIR
revision="post-trained"

export TOKENIZERS_PARALLELISM=false
python eval_cosmos_predict25_lora.py \
  --data_dir $DATA_DIR \
  --output_dir $OUT_DIR \
  --lora_dir $LORA_DIR \
  --revision diffusers/base/$revision \
  --height 432 --width 768 \
  --num_output_frames 93 \
  --num_steps 36 \
  --seed 0
