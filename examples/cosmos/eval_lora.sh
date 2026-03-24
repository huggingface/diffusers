LORA_DIR=YOUR_LORA_WEIGHT_DIR
#LORA_DIR=None # base model
DATA_DIR="gr1_dataset/test"
revision="post-trained"

if [ "$LORA_DIR" != "None" ]; then
  OUTPUT_DIR=$LORA_DIR/results
else
  OUTPUT_DIR=outputs/$revision/results
fi

echo Revision=$revision
echo Data_dir=$DATA_DIR
echo LoRA=$LORA_DIR
echo Out_dir=$OUTPUT_DIR


python_args=(
  --seed 0
  --data_dir $DATA_DIR
  --revision diffusers/base/$revision
  --height 432 --width 768
  --output_dir $OUTPUT_DIR
)

if [ "$LORA_DIR" != "None" ]; then
  python_args+=(--lora_dir $LORA_DIR)
fi

export TOKENIZERS_PARALLELISM=false
python eval_cosmos_predict25_lora.py "${python_args[@]}"
