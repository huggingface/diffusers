<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

<Tip warning={true}>

This is experimental and the API may change in the future.

</Tip>

[LoRA (Low-Rank Adaptation of Large Language Models)](https://hf.co/papers/2106.09685) is a popular and lightweight training technique that significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share. LoRA can also be combined with other training techniques like DreamBooth to speedup training.

<Tip>

LoRA is very versatile and supported for [DreamBooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py), [Kandinsky 2.2](https://github.com/huggingface/diffusers/blob/main/examples/kandinsky2_2/text_to_image/train_text_to_image_lora_decoder.py), [Stable Diffusion XL](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py), [text-to-image](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py), and [Wuerstchen](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_lora_prior.py).

</Tip>

This guide will explore the [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Navigate to the example folder with the training script and install the required dependencies for the script you're using:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/text_to_image
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/text_to_image
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>

🤗 Accelerate is a library for helping you train on multiple GPUs/TPUs or with mixed-precision. It'll automatically configure your training setup based on your hardware and environment. Take a look at the 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) to learn more.

</Tip>

Initialize an 🤗 Accelerate environment:

```bash
accelerate config
```

To setup a default 🤗 Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell, like a notebook, you can use:

```bash
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

<Tip>

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/text_to_image_lora.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training script has many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L85) function. Default values are provided for most parameters that work pretty well, but you can also set your own values in the training command if you'd like.

For example, to increase the number of epochs to train:

```bash
accelerate launch train_text_to_image_lora.py \
  --num_train_epochs=150 \
```

Many of the basic and important parameters are described in the [Text-to-image](text2image#script-parameters) training guide, so this guide just focuses on the LoRA relevant parameters:

- `--rank`: the number of low-rank matrices to train
- `--learning_rate`: the default learning rate is 1e-4, but with LoRA, you can use a higher learning rate

## Training script

The dataset preprocessing code and training loop are found in the [`main()`](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L371) function, and if you need to adapt the training script, this is where you'll make your changes.

As with the script parameters, a walkthrough of the training script is provided in the [Text-to-image](text2image#training-script) training guide. Instead, this guide takes a look at the LoRA relevant parts of the script.

The script begins by adding the [new LoRA weights](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L447) to the attention layers. This involves correctly configuring the weight size for each block in the UNet. You'll see the `rank` parameter is used to create the [`~models.attention_processor.LoRAAttnProcessor`]:

```py
lora_attn_procs = {}
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]

    lora_attn_procs[name] = LoRAAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        rank=args.rank,
    )

unet.set_attn_processor(lora_attn_procs)
lora_layers = AttnProcsLayers(unet.attn_processors)
```

The [optimizer](https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L519) is initialized with the `lora_layers` because these are the only weights that'll be optimized:

```py
optimizer = optimizer_cls(
    lora_layers.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Aside from setting up the LoRA layers, the training script is more or less the same as train_text_to_image.py!

## Launch the script

Once you've made all your changes or you're okay with the default configuration, you're ready to launch the training script! 🚀

Let's train on the [Pokémon BLIP captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset to generate our yown Pokémon. Set the environment variables `MODEL_NAME` and `DATASET_NAME` to the model and dataset respectively. You should also specify where to save the model in `OUTPUT_DIR`, and the name of the model to save to on the Hub with `HUB_MODEL_ID`. The script creates and saves the following files to your repository:

- saved model checkpoints
- `pytorch_lora_weights.safetensors` (the trained LoRA weights)

If you're training on more than one GPU, add the `--multi_gpu` parameter to the `accelerate launch` command.

<Tip warning={true}>

A full training run takes ~5 hours on a 2080 Ti GPU with 11GB of VRAM.

</Tip>

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337
```

Once training has been completed, you can use your model for inference:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/lora/model", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A pokemon with blue eyes").images[0]
```

## Next steps

Congratulations on training a new model with LoRA! To learn more about how to use your new model, the following guides may be helpful:

- Learn how to [load different LoRA formats](../using-diffusers/loading_adapters#LoRA) trained using community trainers like Kohya and TheLastBen.
- Learn how to use and [combine multiple LoRA's](../tutorials/using_peft_for_inference) with PEFT for inference.