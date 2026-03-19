revision='post-trained'
data_dir='dream_gen_benchmark/gr1_object'

lora_dir=YOUR_LORA_WEIGHT_DIR
#lora_dir=None

if [ "$lora_dir" != "None" ]; then
  output_dir=$lora_dir/results
else
  output_dir=outputs/$revision/results
fi

echo Revision=$revision
echo Data_dir=$data_dir
echo LoRA=$lora_dir
echo Out_dir=$output_dir


python_args=(
  --seed 0
  --data_dir $data_dir
  --revision diffusers/base/$revision
  --height 432 --width 768
  --output_dir $output_dir
)

if [ "$lora_dir" != "None" ]; then
  python_args+=(--lora_dir $lora_dir)
fi

export TOKENIZERS_PARALLELISM=false
python ./scripts/eval_cosmos_predict25_lora.py "${python_args[@]}"
