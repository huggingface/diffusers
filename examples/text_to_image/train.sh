###stable_diffusion.sh PSEUDO###
if [[ "$@" == *"--task"* ]]; then # --env_name: conda env name
  env_name="$(echo "$@" | sed -n 's/.*--task \([^ ]*\).*/\1/p')"
  echo "env_name: $env_name"
fi

if [[ "$@" == *"--model_name"* ]]; then
  model_name="$(echo "$@" | sed -n 's/.*--model_name \([^ ]*\).*/\1/p')"
fi

if [[ "$@" == *"--batch_size"* ]]; then 
  batch_size="$(echo "$@" | sed -n 's/.*--batch_size \([^ ]*\).*/\1/p')"
fi

output_dir=outputs
if [[ "$@" == *"--output_dir"* ]]; then 
  output_dir="$(echo "$@" | sed -n 's/.*--output_dir \([^ ]*\).*/\1/p')"
fi

current_dir=$(pwd)
log_dir=${current_dir}/logs/${model_name}.json
if [[ "$@" == *"--log_dir"* ]]; then 
  log_dir="$(echo "$@" | sed -n 's/.*--log_dir \([^ ]*\).*/\1/p')"
fi

# Check if $batch_size is provided , if not use default value of 1
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=1
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 train_text_to_image.py \
  --pretrained_model_name_or_path ${model_name} \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size  ${batch_size}\
  --gradient_accumulation_steps=4 \
  --max_train_steps=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir ${output_dir} \
  --log_dir "${log_dir}" \