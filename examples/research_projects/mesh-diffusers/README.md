# Mesh Diffusers

This script was added by @lk-wq .


### Training in model parallel mode

Add the flag --model_parallel to train the Unet in a model parallel mode, taking advantage of an 8 device mesh on TPUs. The code below has been tested on TPU v3-8.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python3 train_text_to_image_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=16 \
  --mixed_precision="bf16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-pokemon-model" \
  --from_pt \
  --model_parallel
```
