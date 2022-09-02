# Textual Inversion fine-tuning example

### Installing the dependencies

Before running the scipts, make sure to install the library's training dependencies:

```bash
pip install diffusers[training] accelerate transformers
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```


### Cat toy example

First, download 3-4 images from [here](https://drive.google.com/drive/folders/1fmJMs25nxS_rSNqS5hTcRdLem_YQXbq5) and save them in a directory. This will be our training data.


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-images"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME --use_auth_token \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"
```

A full training run takes ~1 hour on one V100 GPU.