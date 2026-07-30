<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Wuerstchen

[Wuerstchen](https://hf.co/papers/2306.00637) 模型通过将潜在空间压缩 42 倍，在不影响图像质量的情况下大幅降低计算成本并加速推理。在训练过程中，Wuerstchen 使用两个模型（VQGAN + 自动编码器）来压缩潜在表示，然后第三个模型（文本条件潜在扩散模型）在这个高度压缩的空间上进行条件化以生成图像。

为了将先验模型放入 GPU 内存并加速训练，尝试分别启用 `gradient_accumulation_steps`、`gradient_checkpointing` 和 `mixed_precision`。

本指南探讨 [train_text_to_image_prior.py](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) 脚本，帮助您更熟悉它，以及如何根据您的用例进行适配。

在运行脚本之前，请确保从源代码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后导航到包含训练脚本的示例文件夹，并安装脚本所需的依赖项：

```bash
cd examples/wuerstchen/text_to_image
pip install -r requirements.txt
```

> [!TIP]
> 🤗 Accelerate 是一个帮助您在多个 GPU/TPU 上或使用混合精度进行训练的库。它会根据您的硬件和环境自动配置训练设置。查看 🤗 Accelerate [快速入门](https://huggingface.co/docs/accelerate/quicktour) 以了解更多信息。

初始化一个 🤗 Accelerate 环境：

```bash
accelerate config
```

要设置一个默认的 🤗 Accelerate 环境而不选择任何配置：

```bash
accelerate config default
```

或者，如果您的环境不支持交互式 shell，例如笔记本，您可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如果您想在自己的数据集上训练模型，请查看 [创建训练数据集](create_dataset) 指南，了解如何创建与训练脚本兼容的数据集。

> [!TIP]
> 以下部分重点介绍了训练脚本中对于理解如何修改它很重要的部分，但并未涵盖 [脚本](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/train_text_to_image_prior.py) 的详细信息。如果您有兴趣了解更多，请随时阅读脚本，并告诉我们您是否有任何问题或疑虑。

## 脚本参数

训练脚本提供了许多参数来帮助您自定义训练运行。所有参数及其描述都可以在 [`parse_args()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L192) 函数中找到。它为每个参数提供了默认值，例如训练批次大小和学习率，但如果您愿意，也可以在训练命令中设置自己的值。

例如，要使用 fp16 格式的混合精度加速训练，请在训练命令中添加 `--mixed_precision` 参数：

```bash
accelerate launch train_text_to_image_prior.py \
  --mixed_precision="fp16"
```

大多数参数与 [文本到图像](text2image#script-parameters) 训练指南中的参数相同，因此让我们直接深入 Wuerstchen 训练脚本！

## 训练脚本

训练脚本也与 [文本到图像](text2image#training-script) 训练指南类似，但已修改以支持 Wuerstchen。本指南重点介绍 Wuerstchen 训练脚本中独特的代码。

[`main()`](https://github.com/huggingface/diffusers/blob/6e68c71503682c8693cb5b06a4da4911dfd655ee/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L441) 函数首先初始化图像编码器 - 一个 [EfficientNet](https://github.com/huggingface/diffusers/blob/main/examples/wuerstchen/text_to_image/modeling_efficient_net_encoder.py) - 以及通常的调度器和分词器。

```py
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    pretrained_checkpoint_file = hf_hub_download("dome272/wuerstchen", filename="model_v2_stage_b.pt")
    state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
    image_encoder = EfficientNetEncoder()
    image_encoder.load_state_dict(state_dict["effnet_state_dict"])
    image_encoder.eval()
```

您还将加载 [`WuerstchenPrior`] 模型以进行优化。

```py
prior = WuerstchenPrior.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="prior")

optimizer = optimizer_cls(
    prior.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```

接下来，您将对图像应用一些 [transforms](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656) 并对标题进行 [tokenize](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L637)：

```py
def preprocess_train(examples):
    images = [image.conver
t("RGB") for image in examples[image_column]]
    examples["effnet_pixel_values"] = [effnet_transforms(image) for image in images]
    examples["text_input_ids"], examples["text_mask"] = tokenize_captions(examples)
    return examples
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/65ef7a0c5c594b4f84092e328fbdd73183613b30/examples/wuerstchen/text_to_image/train_text_to_image_prior.py#L656)处理使用`EfficientNetEncoder`将图像压缩到潜在空间，向潜在表示添加噪声，并使用[`WuerstchenPrior`]模型预测噪声残差。

```py
pred_noise = prior(noisy_latents, timesteps, prompt_embeds)
```

如果您想了解更多关于训练循环的工作原理，请查看[理解管道、模型和调度器](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

## 启动脚本

一旦您完成了所有更改或对默认配置满意，就可以启动训练脚本了！🚀

设置`DATASET_NAME`环境变量为Hub中的数据集名称。本指南使用[Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)数据集，但您也可以创建和训练自己的数据集（参见[创建用于训练的数据集](create_dataset)指南）。

> [!TIP]
> 要使用Weights & Biases监控训练进度，请在训练命令中添加`--report_to=wandb`参数。您还需要在训练命令中添加`--validation_prompt`以跟踪结果。这对于调试模型和查看中间结果非常有用。

```bash
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch  train_text_to_image_prior.py \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompts="A robot naruto, 4k photo" \
  --report_to="wandb" \
  --push_to_hub \
  --output_dir="wuerstchen-prior-naruto-model"
```

训练完成后，您可以使用新训练的模型进行推理！

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipeline = AutoPipelineForText2Image.from_pretrained("path/to/saved/model", dtype=torch.float16).to("cuda")

caption = "A cute bird naruto holding a shield"
images = pipeline(
    caption,
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```

## 下一步

恭喜您训练了一个Wuerstchen模型！要了解更多关于如何使用您的新模型的信息，请参
以下内容可能有所帮助：

- 查看 [Wuerstchen](../api/pipelines/wuerstchen#text-to-image-generation) API 文档，了解更多关于如何使用该管道进行文本到图像生成及其限制的信息。