## Textual Inversion fine-tuning example for SDXL

The `textual_inversion.py` do not support training stable-diffusion-XL as it has two text encoders, you can training SDXL by the following command:
```
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATA_DIR="./cat"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --mixed_precision="bf16" \
  --resolution=1024 \
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

We only enabled training the first text encoder because of the precision issue, we will enable training the second text encoder once we fixed the problem.