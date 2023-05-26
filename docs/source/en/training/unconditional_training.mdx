<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Unconditional image generation

Unconditional image generation is not conditioned on any text or images, unlike text- or image-to-image models. It only generates images that resemble its training data distribution.

<iframe
	src="https://stevhliu-ddpm-butterflies-128.hf.space"
	frameborder="0"
	width="850"
	height="550"
></iframe>


This guide will show you how to train an unconditional image generation model on existing datasets as well as your own custom dataset. All the training scripts for unconditional image generation can be found [here](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation) if you're interested in learning more about the training details.

Before running the script, make sure you install the library's training dependencies:

```bash
pip install diffusers[training] accelerate datasets
```

Next, initialize an 🤗 [Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

To setup a default 🤗 Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell like a notebook, you can use:

```bash
from accelerate.utils import write_basic_config

write_basic_config()
```

## Upload model to Hub

You can upload your model on the Hub by adding the following argument to the training script:

```bash
--push_to_hub
```

## Save and load checkpoints

It is a good idea to regularly save checkpoints in case anything happens during training. To save a checkpoint, pass the following argument to the training script:

```bash
--checkpointing_steps=500
```

The full training state is saved in a subfolder in the `output_dir` every 500 steps, which allows you to load a checkpoint and resume training if you pass the `--resume_from_checkpoint` argument to the training script:

```bash
--resume_from_checkpoint="checkpoint-1500"
```

## Finetuning

You're ready to launch the [training script](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py) now! Specify the dataset name to finetune on with the `--dataset_name` argument and then save it to the path in `--output_dir`. To use your own dataset, take a look at the [Create a dataset for training](create_dataset) guide.

The training script creates and saves a `diffusion_pytorch_model.bin` file in your repository.

<Tip>

💡 A full training run takes 2 hours on 4xV100 GPUs.

</Tip>

For example, to finetune on the [Oxford Flowers](https://huggingface.co/datasets/huggan/flowers-102-categories) dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
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

<div class="flex justify-center">
    <img src="https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png"/>
</div>

Or if you want to train your model on the [Pokemon](https://huggingface.co/datasets/huggan/pokemon) dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="huggan/pokemon" \
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

<div class="flex justify-center">
    <img src="https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png"/>
</div>

### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger="wandb" \
  --push_to_hub
```