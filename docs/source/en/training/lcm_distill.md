<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Latent Consistency Distillation

[Latent Consistency Models (LCMs)](https://hf.co/papers/2310.04378) are able to generate high-quality images in just a few steps, representing a big leap forward because many pipelines require at least 25+ steps. LCMs are produced by applying the latent consistency distillation method to any Stable Diffusion model. This method works by applying *one-stage guided distillation* to the latent space, and incorporating a *skipping-step* method to consistently skip timesteps to accelerate the distillation process (refer to section 4.1, 4.2, and 4.3 of the paper for more details).

If you're training on a GPU with limited vRAM, try enabling `gradient_checkpointing`, `gradient_accumulation_steps`, and `mixed_precision` to reduce memory-usage and speedup training. You can reduce your memory-usage even more by enabling memory-efficient attention with [xFormers](../optimization/xformers) and [bitsandbytes'](https://github.com/TimDettmers/bitsandbytes) 8-bit optimizer.

This guide will explore the [train_lcm_distill_sd_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sd_wds.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then navigate to the example folder containing the training script and install the required dependencies for the script you're using:

```bash
cd examples/consistency_distillation
pip install -r requirements.txt
```

> [!TIP]
> 🤗 Accelerate is a library for helping you train on multiple GPUs/TPUs or with mixed-precision. It'll automatically configure your training setup based on your hardware and environment. Take a look at the 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) to learn more.

Initialize an 🤗 Accelerate environment (try enabling `torch.compile` to significantly speedup training):

```bash
accelerate config
```

To setup a default 🤗 Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell, like a notebook, you can use:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

## Script parameters

> [!TIP]
> The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sd_wds.py) and let us know if you have any questions or concerns.

The training script provides many parameters to help you customize your training run. All of the parameters and their descriptions are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L419) function. This function provides default values for each parameter, such as the training batch size and learning rate, but you can also set your own values in the training command if you'd like.

For example, to speedup training with mixed precision using the fp16 format, add the `--mixed_precision` parameter to the training command:

```bash
accelerate launch train_lcm_distill_sd_wds.py \
  --mixed_precision="fp16"
```

Most of the parameters are identical to the parameters in the [Text-to-image](text2image#script-parameters) training guide, so you'll focus on the parameters that are relevant to latent consistency distillation in this guide.

- `--pretrained_teacher_model`: the path to a pretrained latent diffusion model to use as the teacher model
- `--pretrained_vae_model_name_or_path`: path to a pretrained VAE; the SDXL VAE is known to suffer from numerical instability, so this parameter allows you to specify an alternative VAE (like this [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)) by madebyollin which works in fp16)
- `--w_min` and `--w_max`: the minimum and maximum guidance scale values for guidance scale sampling
- `--num_ddim_timesteps`: the number of timesteps for DDIM sampling
- `--loss_type`: the type of loss (L2 or Huber) to calculate for latent consistency distillation; Huber loss is generally preferred because it's more robust to outliers
- `--huber_c`: the Huber loss parameter

## Training script

The training script starts by creating a dataset class - [`Text2ImageDataset`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L141) - for preprocessing the images and creating a training dataset.

```py
def transform(example):
    image = example["image"]
    image = TF.resize(image, resolution, interpolation=transforms.InterpolationMode.BILINEAR)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    image = TF.crop(image, c_top, c_left, resolution, resolution)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])

    example["image"] = image
    return example
```

For improved performance on reading and writing large datasets stored in the cloud, this script uses the [WebDataset](https://github.com/webdataset/webdataset) format to create a preprocessing pipeline to apply transforms and create a dataset and dataloader for training. Images are processed and fed to the training loop without having to download the full dataset first.

```py
processing_pipeline = [
    wds.decode("pil", handler=wds.ignore_and_continue),
    wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
    wds.map(filter_keys({"image", "text"})),
    wds.map(transform),
    wds.to_tuple("image", "text"),
]
```

In the [`main()`](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L768) function, all the necessary components like the noise scheduler, tokenizers, text encoders, and VAE are loaded. The teacher UNet is also loaded here and then you can create a student UNet from the teacher UNet. The student UNet is updated by the optimizer during training.

```py
teacher_unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_teacher_model, subfolder="unet", revision=args.teacher_revision
)

unet = UNet2DConditionModel(**teacher_unet.config)
unet.load_state_dict(teacher_unet.state_dict(), strict=False)
unet.train()
```

Now you can create the [optimizer](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L979) to update the UNet parameters:

```py
optimizer = optimizer_class(
    unet.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

Create the [dataset](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L994):

```py
dataset = Text2ImageDataset(
    train_shards_path_or_url=args.train_shards_path_or_url,
    num_train_examples=args.max_train_samples,
    per_gpu_batch_size=args.train_batch_size,
    global_batch_size=args.train_batch_size * accelerator.num_processes,
    num_workers=args.dataloader_num_workers,
    resolution=args.resolution,
    shuffle_buffer_size=1000,
    pin_memory=True,
    persistent_workers=True,
)
train_dataloader = dataset.train_dataloader
```

Next, you're ready to setup the [training loop](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1049) and implement the latent consistency distillation method (see Algorithm 1 in the paper for more details). This section of the script takes care of adding noise to the latents, sampling and creating a guidance scale embedding, and predicting the original image from the noise.

```py
pred_x_0 = predicted_origin(
    noise_pred,
    start_timesteps,
    noisy_model_input,
    noise_scheduler.config.prediction_type,
    alpha_schedule,
    sigma_schedule,
)

model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
```

It gets the [teacher model predictions](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1172) and the [LCM predictions](https://github.com/huggingface/diffusers/blob/3b37488fa3280aed6a95de044d7a42ffdcb565ef/examples/consistency_distillation/train_lcm_distill_sd_wds.py#L1209) next, calculates the loss, and then backpropagates it to the LCM.

```py
if args.loss_type == "l2":
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
elif args.loss_type == "huber":
    loss = torch.mean(
        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
    )
```

If you want to learn more about how the training loop works, check out the [Understanding pipelines, models and schedulers tutorial](../using-diffusers/write_own_pipeline) which breaks down the basic pattern of the denoising process.

## Launch the script

Now you're ready to launch the training script and start distilling!

For this guide, you'll use the `--train_shards_path_or_url` to specify the path to the [Conceptual Captions 12M](https://github.com/google-research-datasets/conceptual-12m) dataset stored on the Hub [here](https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset). Set the `MODEL_DIR` environment variable to the name of the teacher model and `OUTPUT_DIR` to where you want to save the model.

```bash
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub
```

Once training is complete, you can use your new LCM for inference.

```py
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained("your-username/your-model", torch_dtype=torch.float16, variant="fp16")
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipeline.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipeline.to("cuda")

prompt = "sushi rolls in the form of panda heads, sushi platter"

image = pipeline(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]
```

## LoRA

LoRA is a training technique for significantly reducing the number of trainable parameters. As a result, training is faster and it is easier to store the resulting weights because they are a lot smaller (~100MBs). Use the [train_lcm_distill_lora_sd_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py) or [train_lcm_distill_lora_sdxl.wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl_wds.py) script to train with LoRA.

The LoRA training script is discussed in more detail in the [LoRA training](lora) guide.

## Stable Diffusion XL

Stable Diffusion XL (SDXL) is a powerful text-to-image model that generates high-resolution images, and it adds a second text-encoder to its architecture. Use the [train_lcm_distill_sdxl_wds.py](https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_sdxl_wds.py) script to train a SDXL model with LoRA.

The SDXL training script is discussed in more detail in the [SDXL training](sdxl) guide.

## Next steps

Congratulations on distilling a LCM model! To learn more about LCM, the following may be helpful:

- Learn how to use [LCMs for inference](../using-diffusers/inference_with_lcm) for text-to-image, image-to-image, and with LoRA checkpoints.
- Read the [SDXL in 4 steps with Latent Consistency LoRAs](https://huggingface.co/blog/lcm_lora) blog post to learn more about SDXL LCM-LoRA's for super fast inference, quality comparisons, benchmarks, and more.
