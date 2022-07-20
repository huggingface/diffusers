## Training examples

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

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/173855866-5628989f-856b-4725-a944-d6c09490b2df.png" width="500" />


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

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/173856733-4f117f8c-97bd-4f51-8002-56b488c96df9.png" width="500" />
