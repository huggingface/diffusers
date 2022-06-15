## Training examples

### Flowers DDPM 

The command to train a DDPM UNet model on the Oxford Flowers dataset:

```bash
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  train_ddpm.py \
  --dataset="huggan/flowers-102-categories" \
  --resolution=64 \
  --output_path="flowers-ddpm" \
  --batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --lr=1e-4 \
  --warmup_steps=500 \
  --mixed_precision=no
```

A full ltraining run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/173855866-5628989f-856b-4725-a944-d6c09490b2df.png" width="500" />


### Pokemon DDPM 

The command to train a DDPM UNet model on the Pokemon dataset:

```bash
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  train_ddpm.py \
  --dataset="huggan/pokemon" \
  --resolution=64 \
  --output_path="flowers-ddpm" \
  --batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --lr=1e-4 \
  --warmup_steps=500 \
  --mixed_precision=no
```

A full ltraining run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/173856733-4f117f8c-97bd-4f51-8002-56b488c96df9.png" width="500" />
