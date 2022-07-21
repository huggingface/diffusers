## Training examples

### Installing the dependencies

Before running the scipts, make sure to install the library's training dependencies:

```bash
pip install diffusers[training] accelerate datasets
```

### Unconditional Flowers  

The command to train a DDPM UNet model on the Oxford Flowers dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset="huggan/flowers-102-categories" \
  --resolution=64 \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```
An example trained model: https://huggingface.co/anton-l/ddpm-ema-flowers-64

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png" width="700" />


### Unconditional Pokemon 

The command to train a DDPM UNet model on the Pokemon dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset="huggan/pokemon" \
  --resolution=64 \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```
An example trained model: https://huggingface.co/anton-l/ddpm-ema-pokemon-64

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png" width="700" />
