<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# InstructPix2Pix

[InstructPix2Pix](https://hf.co/papers/2211.09800) 是一个基于 Stable Diffusion 训练的模型，用于根据人类提供的指令编辑图像。例如，您的提示可以是“将云变成雨天”，模型将相应编辑输入图像。该模型以文本提示（或编辑指令）和输入图像为条件。

本指南将探索 [train_instruct_pix2pix.py](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py) 训练脚本，帮助您熟悉它，以及如何将其适应您自己的用例。

在运行脚本之前，请确保从源代码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后导航到包含训练脚本的示例文件夹，并安装脚本所需的依赖项：

```bash
cd examples/instruct_pix2pix
pip install -r requirements.txt
```

> [!TIP]
> 🤗 Accelerate 是一个库，用于帮助您在多个 GPU/TPU 上或使用混合精度进行训练。它将根据您的硬件和环境自动配置训练设置。查看 🤗 Accelerate [快速导览](https://huggingface.co/docs/accelerate/quicktour) 以了解更多信息。

初始化一个 🤗 Accelerate 环境：

```bash
accelerate config
```

要设置一个默认的 🤗 Accelerate 环境，无需选择任何配置：

```bash
accelerate config default
```

或者，如果您的环境不支持交互式 shell，例如笔记本，您可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如果您想在自己的数据集上训练模型，请查看 [创建用于训练的数据集](create_dataset) 指南，了解如何创建与训练脚本兼容的数据集。

> [!TIP]
> 以下部分重点介绍了训练脚本中对于理解如何修改它很重要的部分，但并未详细涵盖脚本的每个方面。如果您有兴趣了解更多，请随时阅读 [脚本](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py)，并告诉我们如果您有任何问题或疑虑。

## 脚本参数

训练脚本有许多参数可帮助您自定义训练运行。所有
参数及其描述可在 [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L65) 函数中找到。大多数参数都提供了默认值，这些值效果相当不错，但如果您愿意，也可以在训练命令中设置自己的值。

例如，要增加输入图像的分辨率：

```bash
accelerate launch train_instruct_pix2pix.py \
  --resolution=512 \
```

许多基本和重要的参数在 [文本到图像](text2image#script-parameters) 训练指南中已有描述，因此本指南仅关注与 InstructPix2Pix 相关的参数：

- `--original_image_column`：编辑前的原始图像
- `--edited_image_column`：编辑后的图像
- `--edit_prompt_column`：编辑图像的指令
- `--conditioning_dropout_prob`：训练期间编辑图像和编辑提示的 dropout 概率，这为一种或两种条件输入启用了无分类器引导（CFG）

## 训练脚本

数据集预处理代码和训练循环可在 [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L374) 函数中找到。这是您将修改训练脚本以适应自己用例的地方。

与脚本参数类似，[文本到图像](text2image#training-script) 训练指南提供了训练脚本的逐步说明。相反，本指南将查看脚本中与 InstructPix2Pix 相关的部分。

脚本首先修改 UNet 的第一个卷积层中的 [输入通道数](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L445)，以适应 InstructPix2Pix 的额外条件图像：

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

这些 UNet 参数由优化器 [更新](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L545C1-L551C6)：

```py
optimizer = optimizer_cls(
    unet.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

接下来，编辑后的图像和编辑指令被 [预处理](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L624)并被[tokenized](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L610C24-L610C24)。重要的是，对原始图像和编辑后的图像应用相同的图像变换。

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

最后，在[训练循环](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/instruct_pix2pix/train_instruct_pix2pix.py#L730)中，它首先将编辑后的图像编码到潜在空间：

```py
latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

然后，脚本对原始图像和编辑指令嵌入应用 dropout 以支持 CFG（Classifier-Free Guidance）。这使得模型能够调节编辑指令和原始图像对编辑后图像的影响。

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

差不多就是这样了！除了这里描述的不同之处，脚本的其余部分与[文本到图像](text2image#training-script)训练脚本非常相似，所以请随意查看以获取更多细节。如果您想了解更多关于训练循环如何工作的信息，请查看[理解管道、模型和调度器](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

## 启动脚本

一旦您对脚本的更改感到满意，或者如果您对默认配置没问题，您
准备好启动训练脚本！🚀

本指南使用 [fusing/instructpix2pix-1000-samples](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples) 数据集，这是 [原始数据集](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) 的一个较小版本。您也可以创建并使用自己的数据集（请参阅 [创建用于训练的数据集](create_dataset) 指南）。

将 `MODEL_NAME` 环境变量设置为模型名称（可以是 Hub 上的模型 ID 或本地模型的路径），并将 `DATASET_ID` 设置为 Hub 上数据集的名称。脚本会创建并保存所有组件（特征提取器、调度器、文本编码器、UNet 等）到您的仓库中的一个子文件夹。

> [!TIP]
> 为了获得更好的结果，尝试使用更大的数据集进行更长时间的训练。我们只在较小规模的数据集上测试过此训练脚本。
>
> <br>
>
> 要使用 Weights and Biases 监控训练进度，请将 `--report_to=wandb` 参数添加到训练命令中，并使用 `--val_image_url` 指定验证图像，使用 `--validation_prompt` 指定验证提示。这对于调试模型非常有用。

如果您在多个 GPU 上训练，请将 `--multi_gpu` 参数添加到 `accelerate launch` 命令中。

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

训练完成后，您可以使用您的新 InstructPix2Pix 进行推理：

```py
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("your_cool_model", dtype=torch.float16).to("cuda")
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

您应该尝试不同的 `num_inference_steps`、`image_guidance_scale` 和 `guidance_scale` 值，以查看它们如何影响推理速度和质量。指导比例参数
这些参数尤其重要，因为它们控制原始图像和编辑指令对编辑后图像的影响程度。

## Stable Diffusion XL

Stable Diffusion XL (SDXL) 是一个强大的文本到图像模型，能够生成高分辨率图像，并在其架构中添加了第二个文本编码器。使用 [`train_instruct_pix2pix_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py) 脚本来训练 SDXL 模型以遵循图像编辑指令。

SDXL 训练脚本在 [SDXL 训练](sdxl) 指南中有更详细的讨论。

## 后续步骤

恭喜您训练了自己的 InstructPix2Pix 模型！🥳 要了解更多关于该模型的信息，可能有助于：

- 阅读 [Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd) 博客文章，了解更多我们使用 InstructPix2Pix 进行的一些实验、数据集准备以及不同指令的结果。