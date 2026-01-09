<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 文生图

> [!WARNING]
> 文生图训练脚本目前处于实验阶段，容易出现过拟合和灾难性遗忘等问题。建议尝试不同超参数以获得最佳数据集适配效果。

Stable Diffusion 等文生图模型能够根据文本提示生成对应图像。

模型训练对硬件要求较高，但启用 `gradient_checkpointing` 和 `mixed_precision` 后，可在单块24GB显存GPU上完成训练。如需更大批次或更快训练速度，建议使用30GB以上显存的GPU设备。通过启用 [xFormers](../optimization/xformers) 内存高效注意力机制可降低显存占用。JAX/Flax 训练方案也支持TPU/GPU高效训练，但不支持梯度检查点、梯度累积和xFormers。使用Flax训练时建议配备30GB以上显存GPU或TPU v3。

本指南将详解 [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) 训练脚本，助您掌握其原理并适配自定义需求。

运行脚本前请确保已从源码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

然后进入包含训练脚本的示例目录，安装对应依赖：

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

> [!TIP]
> 🤗 Accelerate 是支持多GPU/TPU训练和混合精度的工具库，能根据硬件环境自动配置训练参数。参阅 🤗 Accelerate [快速入门](https://huggingface.co/docs/accelerate/quicktour) 了解更多。

初始化 🤗 Accelerate 环境：

```bash
accelerate config
```

要创建默认配置环境（不进行交互式选择）：

```bash
accelerate config default
```

若环境不支持交互式shell（如notebook），可使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如需在自定义数据集上训练，请参阅 [创建训练数据集](create_dataset) 指南了解如何准备适配脚本的数据集。

## 脚本参数

> [!TIP]
> 以下重点介绍脚本中影响训练效果的关键参数，如需完整参数说明可查阅 [脚本源码](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)。如有疑问欢迎反馈。

训练脚本提供丰富参数供自定义训练流程，所有参数及说明详见 [`parse_args()`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L193) 函数。该函数为每个参数提供默认值（如批次大小、学习率等），也可通过命令行参数覆盖。

例如使用fp16混合精度加速训练：

```bash
accelerate launch train_text_to_image.py \
  --mixed_precision="fp16"
```

基础重要参数包括：

- `--pretrained_model_name_or_path`: Hub模型名称或本地预训练模型路径
- `--dataset_name`: Hub数据集名称或本地训练数据集路径
- `--image_column`: 数据集中图像列名
- `--caption_column`: 数据集中文本列名
- `--output_dir`: 模型保存路径
- `--push_to_hub`: 是否将训练模型推送至Hub
- `--checkpointing_steps`: 模型检查点保存步数；训练中断时可添加 `--resume_from_checkpoint` 从该检查点恢复训练

### Min-SNR加权策略

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略通过重新平衡损失函数加速模型收敛。训练脚本支持预测 `epsilon`（噪声）或 `v_prediction`，而Min-SNR兼容两种预测类型。该策略仅限PyTorch版本，Flax训练脚本不支持。

添加 `--snr_gamma` 参数并设为推荐值5.0：

```bash
accelerate launch train_text_to_image.py \
  --snr_gamma=5.0
```

可通过此 [Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr) 报告比较不同 `snr_gamma` 值的损失曲面。小数据集上Min-SNR效果可能不如大数据集显著。

## 训练脚本解析

数据集预处理代码和训练循环位于 [`main()`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L490) 函数，自定义修改需在此处进行。

`train_text_to_image` 脚本首先 [加载调度器](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L543) 和分词器，此处可替换其他调度器：

```py
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
```

接着 [加载UNet模型](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L619)：

```py
load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
model.register_to_config(**load_model.config)

model.load_state_dict(load_model.state_dict())
```

随后对数据集的文本和图像列进行预处理。[`tokenize_captions`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L724) 函数处理文本分词，[`train_transforms`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L742) 定义图像增强策略，二者集成于 `preprocess_train`：

```py
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L878) 处理剩余流程：图像编码为潜空间、添加噪声、计算文本嵌入条件、更新模型参数、保存并推送模型至Hub。想深入了解训练循环原理，可参阅 [理解管道、模型与调度器](../using-diffusers/write_own_pipeline) 教程，该教程解析了去噪过程的核心逻辑。

## 启动脚本

完成所有配置后，即可启动训练脚本！🚀

<hfoptions id="training-inference">
<hfoption id="PyTorch">

以 [火影忍者BLIP标注数据集](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) 为例训练生成火影角色。设置环境变量 `MODEL_NAME` 和 `dataset_name` 指定模型和数据集（Hub或本地路径）。多GPU训练需在 `accelerate launch` 命令中添加 `--multi_gpu` 参数。

> [!TIP]
> 使用本地数据集时，设置 `TRAIN_DIR` 和 `OUTPUT_DIR` 环境变量为数据集路径和模型保存路径。

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```

</hfoption>
<hfoption id="Flax">

Flax训练方案在TPU/GPU上效率更高（由 [@duongna211](https://github.com/duongna21) 实现），TPU性能更优但GPU表现同样出色。

设置环境变量 `MODEL_NAME` 和 `dataset_name` 指定模型和数据集（Hub或本地路径）。

> [!TIP]
> 使用本地数据集时，设置 `TRAIN_DIR` 和 `OUTPUT_DIR` 环境变量为数据集路径和模型保存路径。

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

python train_text_to_image_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```

</hfoption>
</hfoptions>

训练完成后，即可使用新模型进行推理：

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("path/to/saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline(prompt="yoda").images[0]
image.save("yoda-naruto.png")
```

</hfoption>
<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path/to/saved_model", dtype=jax.numpy.bfloat16)

prompt = "yoda naruto"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# 分片输入和随机数
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("yoda-naruto.png")
```

</hfoption>
</hfoptions>

## 后续步骤

恭喜完成文生图模型训练！如需进一步使用模型，以下指南可能有所帮助：

- 了解如何加载 [LoRA权重](../using-diffusers/loading_adapters#LoRA) 进行推理（如果训练时使用了LoRA）
- 在 [文生图](../using-diffusers/conditional_image_generation) 任务指南中，了解引导尺度等参数或提示词加权等技术如何控制生成效果