cd /nas/thuchk/repos/diffusers/examples/textual_inversion/
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./cat"

if [ -d "./cat" ] && [ "$(ls -A ./cat)" ]
then
  echo "./cat exists and is not Empty"
else
  # run a Python script if "./cat" directory does not exist
  python /nas/thuchk/repos/diffusers/examples/textual_inversion/download_data.py
fi

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"
  #   --push_to_hub \