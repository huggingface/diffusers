<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# InstructPix2Pix

[InstructPix2Pix](https://hf.co/papers/2211.09800) is a Stable Diffusion model trained to edit images from human-provided instructions. For example, your prompt can be "turn the clouds rainy" and the model will edit the input image accordingly. This model is conditioned on the text prompt (or editing instruction) and the input image.

This guide will explore the [train_instruct_pix2pix.py](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) training script to help you become familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/instruct_pix2pix
pip install -r requirements.txt
```

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

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training script has many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L65) function. Default values are provided for most parameters that work pretty well, but you can also set your own values in the training command if you'd like.

For example, to increase the resolution of the input image:

```bash
accelerate launch train_instruct_pix2pix.py \
  --resolution=512 \
```

Many of the basic and important parameters are described in the [Text-to-image](text2image#script-parameters) training guide, so this guide just focuses on the relevant parameters for InstructPix2Pix:

- `--original_image_column`: the original image before the edits are made
- `--edited_image_column`: the image after the edits are made
- `--edit_prompt_column`: the instructions to edit the image
- `--conditioning_dropout_prob`: the dropout probability for the edited image and edit prompts during training which enables classifier-free guidance (CFG) for one or both conditioning inputs

## Training script

The dataset preprocessing code and training loop are found in the [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L374) function. This is where you'll make your changes to the training script to adapt it for your own use-case.

As with the script parameters, a walkthrough of the training script is provided in the [Text-to-image](text2image#training-script) training guide. Instead, this guide takes a look at the InstructPix2Pix relevant parts of the script.

The script begins by modifing the [number of input channels](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L445) in the first convolutional layer of the UNet to account for InstructPix2Pix's additional conditioning image:

```py
in_channels = 8
out_channels = unet.conv_in.out_channels
unet.register_to_config(in_channels=in_channels)

with torch.no_grad():
    new_conv_in = nn.Conv2d(
        in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    unet.conv_in = new_conv_in
```

These UNet parameters are [updated](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L545C1-L551C6) by the optimizer:

```py
optimizer = optimizer_cls(
    unet.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Next, the edited images and and edit instructions are [preprocessed](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L624) and [tokenized](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L610C24-L610C24). It is important the same image transformations are applied to the original and edited images.

```py
def preprocess_train(examples):
    preprocessed_images = preprocess_images(examples)

    original_images, edited_images = preprocessed_images.chunk(2)
    original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

    examples["original_pixel_values"] = original_images
    examples["edited_pixel_values"] = edited_images

    captions = list(examples[edit_prompt_column])
    examples["input_ids"] = tokenize_captions(captions)
    return examples
```

Finally, in the [training loop](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L730), it starts by encoding the edited images into latent space:

```py
latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

Then, the script applies dropout to the original image and edit instruction embeddings to support CFG. This is what enables the model to modulate the influence of the edit instruction and original image on the edited image.

```py
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

if args.conditioning_dropout_prob is not None:
    random_p = torch.rand(bsz, device=latents.device, generator=generator)
    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

    image_mask_dtype = original_image_embeds.dtype
    image_mask = 1 - (
        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
    )
    image_mask = image_mask.reshape(bsz, 1, 1, 1)
    original_image_embeds = image_mask * original_image_embeds
```

That's pretty much it! Aside from the differences described here, the rest of the script is very similar to the [Text-to-image](text2image#training-script) training script, so feel free to check it out for more details. If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) tutorial which breaks down the basic pattern of the denoising process.

## Launch the script

Once you're happy with the changes to your script or if you're okay with the default configuration, you're ready to launch the training script! 🚀

This guide uses the [fusing/instructpix2pix-1000-samples](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples) dataset, which is a smaller version of the [original dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered). You can also create and use your own dataset if you'd like (see the [Create a dataset for training](create_dataset) guide).

Set the `MODEL_NAME` environment variable to the name of the model (can be a model id on the Hub or a path to a local model), and the `DATASET_ID` to the name of the dataset on the Hub. The script creates and saves all the components (feature extractor, scheduler, text encoder, UNet, etc.) to a subfolder in your repository.

<Tip>

For better results, try longer training runs with a larger dataset. We've only tested this training script on a smaller-scale dataset.

<br>

To monitor training progress with Weights and Biases, add the `--report_to=wandb` parameter to the training command and specify a validation image with `--val_image_url` and a validation prompt with `--validation_prompt`. This can be really useful for debugging the model.

</Tip>

If you’re training on more than one GPU, add the `--multi_gpu` parameter to the `accelerate launch` command.

```bash
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 \
    --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --push_to_hub
```

After training is finished, you can use your new InstructPix2Pix for inference:

```py
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("your_cool_model", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

image = load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png")
prompt = "add some ducks to the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipeline(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("edited_image.png")
```

You should experiment with different `num_inference_steps`, `image_guidance_scale`, and `guidance_scale` values to see how they affect inference speed and quality. The guidance scale parameters are especially impactful because they control how much the original image and edit instructions affect the edited image.

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [`train_instruct_pix2pix_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py) script to train a SDXL model to follow image editing instructions.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl) guide.

## Next steps

Congratulations on training your own InstructPix2Pix model! 🥳 To learn more about the model, it may be helpful to:

- Read the [Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd) blog post to learn more about some experiments we've done with InstructPix2Pix, dataset preparation, and results for different instructions.