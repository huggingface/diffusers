## Textual Inversion fine-tuning example for SDXL

```sh
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATA_DIR="./cat"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_cat_sdxl"
```

For now, only training of the first text encoder is supported.