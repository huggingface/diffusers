<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版（“许可证”）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定的语言管理权限和限制。
-->

# DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) 是一种训练技术，通过仅训练少数主题或风格的图像来更新整个扩散模型。它通过在提示中关联一个特殊词与示例图像来工作。

如果您在 vRAM 有限的 GPU 上训练，应尝试在训练命令中启用 `gradient_checkpointing` 和 `mixed_precision` 参数。您还可以通过使用 [xFormers](../optimization/xformers) 的内存高效注意力来减少内存占用。JAX/Flax 训练也支持在 TPU 和 GPU 上进行高效训练，但不支持梯度检查点或 xFormers。如果您想使用 Flax 更快地训练，应拥有内存 >30GB 的 GPU。

本指南将探索 [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) 脚本，帮助您更熟悉它，以及如何根据您的用例进行适配。

在运行脚本之前，请确保从源代码安装库：

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

导航到包含训练脚本的示例文件夹，并安装脚本所需的依赖项：

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/dreambooth
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/dreambooth
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

> [!TIP]
> 🤗 Accelerate 是一个库，用于帮助您在多个 GPU/TPU 上或使用混合精度进行训练。它会根据您的硬件和环境自动配置训练设置。查看 🤗 Accelerate [快速入门](https://huggingface.co/docs/accelerate/quicktour) 以了解更多信息。

初始化 🤗 Accelerate 环境：

```bash
accelerate config
```

要设置默认的 🤗 Accelerate 环境而不选择任何配置：

```bash
accelerate config default
```

或者，如果您的环境不支持交互式 shell，例如笔记本，您可以使用：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

最后，如果您想在自己的数据集上训练模型，请查看 [创建用于训练的数据集](create_dataset) 指南，了解如何创建与
训练脚本。

> [!TIP]
> 以下部分重点介绍了训练脚本中对于理解如何修改它很重要的部分，但并未详细涵盖脚本的每个方面。如果您有兴趣了解更多，请随时阅读[脚本](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)，并告诉我们如果您有任何问题或疑虑。

## 脚本参数

> [!WARNING]
> DreamBooth 对训练超参数非常敏感，容易过拟合。阅读 [使用 🧨 Diffusers 训练 Stable Diffusion 与 Dreambooth](https://huggingface.co/blog/dreambooth) 博客文章，了解针对不同主题的推荐设置，以帮助您选择合适的超参数。

训练脚本提供了许多参数来自定义您的训练运行。所有参数及其描述都可以在 [`parse_args()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L228) 函数中找到。参数设置了默认值，这些默认值应该开箱即用效果不错，但如果您愿意，也可以在训练命令中设置自己的值。

例如，要以 bf16 格式进行训练：

```bash
accelerate launch train_dreambooth.py \
    --mixed_precision="bf16"
```

一些基本且重要的参数需要了解和指定：

- `--pretrained_model_name_or_path`: Hub 上的模型名称或预训练模型的本地路径
- `--instance_data_dir`: 包含训练数据集（示例图像）的文件夹路径
- `--instance_prompt`: 包含示例图像特殊单词的文本提示
- `--train_text_encoder`: 是否也训练文本编码器
- `--output_dir`: 保存训练后模型的位置
- `--push_to_hub`: 是否将训练后的模型推送到 Hub
- `--checkpointing_steps`: 模型训练时保存检查点的频率；这在训练因某种原因中断时很有用，您可以通过在训练命令中添加 `--resume_from_checkpoint` 来从该检查点继续训练

### Min-SNR 加权

[Min-SNR](https://huggingface.co/papers/2303.09556) 加权策略可以通过重新平衡损失来帮助训练，以实现更快的收敛。训练脚本支持预测 `epsilon`（噪声）或 `v_prediction`，但 Min-SNR 与两种预测类型都兼容。此加权策略仅由 PyTorch 支持，在 Flax 训练脚本中不可用。

添加 `--snr_gamma` 参数并将其设置为推荐值 5.0：

```bash
accelerate launch train_dreambooth.py \
  --snr_gamma=5.0
```

### 先验保持损失

先验保持损失是一种使用模型自身生成的样本来帮助它学习如何生成更多样化图像的方法。因为这些生成的样本图像属于您提供的图像相同的类别，它们帮助模型 r
etain 它已经学到的关于类别的知识，以及它如何利用已经了解的类别信息来创建新的组合。

- `--with_prior_preservation`: 是否使用先验保留损失
- `--prior_loss_weight`: 控制先验保留损失对模型的影响程度
- `--class_data_dir`: 包含生成的类别样本图像的文件夹路径
- `--class_prompt`: 描述生成的样本图像类别的文本提示

```bash
accelerate launch train_dreambooth.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images" \
  --class_prompt="text prompt describing class"
```

### 训练文本编码器

为了提高生成输出的质量，除了 UNet 之外，您还可以训练文本编码器。这需要额外的内存，并且您需要一个至少有 24GB 显存的 GPU。如果您拥有必要的硬件，那么训练文本编码器会产生更好的结果，尤其是在生成面部图像时。通过以下方式启用此选项：

```bash
accelerate launch train_dreambooth.py \
  --train_text_encoder
```

## 训练脚本

DreamBooth 附带了自己的数据集类：

- [`DreamBoothDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L604): 预处理图像和类别图像，并对提示进行分词以用于训练
- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L738): 生成提示嵌入以生成类别图像

如果您启用了[先验保留损失](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L842)，类别图像在此处生成：

```py
sample_dataset = PromptDataset(args.class_prompt, num_new_images)
sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

sample_dataloader = accelerator.prepare(sample_dataloader)
pipeline.to(accelerator.device)

for example in tqdm(
    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
):
    images = pipeline(example["prompt"]).images
```

接下来是 [`main()`](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L799) 函数，它处理设置训练数据集和训练循环本身。脚本加载 [tokenizer](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L898)、[scheduler 和 models](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L912C1-L912C1)：

```py
# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

# 加载调度器和模型
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)

if model_has_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
else:
    vae = None

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

然后，是时候[创建训练数据集](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1073)和从`DreamBoothDataset`创建DataLoader：

```py
train_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_prompt=args.class_prompt,
    class_num=args.num_class_images,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=args.center_crop,
    encoder_hidden_states=pre_computed_encoder_hidden_states,
    class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
    tokenizer_max_length=args.tokenizer_max_length,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    num_workers=args.dataloader_num_workers,
)
```

最后，[训练循环](https://github.com/huggingface/diffusers/blob/072e00897a7cf4302c347a63ec917b4b8add16d4/examples/dreambooth/train_dreambooth.py#L1151)处理剩余步骤，例如将图像转换为潜在空间、向输入添加噪声、预测噪声残差和计算损失。

如果您想了解更多关于训练循环的工作原理，请查看[理解管道、模型和调度器](../using-diffusers/write_own_pipeline)教程，该教程分解了去噪过程的基本模式。

## 启动脚本

您现在准备好启动训练脚本了！🚀

对于本指南，您将下载一些[狗的图片](https://huggingface.co/datasets/diffusers/dog-example)的图像并将它们存储在一个目录中。但请记住，您可以根据需要创建和使用自己的数据集（请参阅[创建用于训练的数据集](create_dataset)指南）。

```py
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

设置环境变量 `MODEL_NAME` 为 Hub 上的模型 ID 或本地模型路径，`INSTANCE_DIR` 为您刚刚下载狗图像的路径，`OUTPUT_DIR` 为您想保存模型的位置。您将使用 `sks` 作为特殊词来绑定训练。

如果您有兴趣跟随训练过程，可以定期保存生成的图像作为训练进度。将以下参数添加到训练命令中：

```bash
--validation_prompt="a photo of a sks dog"
--num_validation_images=4
--validation_steps=100
```

在启动脚本之前，还有一件事！根据您拥有的 GPU，您可能需要启用某些优化来训练 DreamBooth。

<hfoptions id="gpu-select">
<hfoption id="16GB">

在 16GB GPU 上，您可以使用 bitsandbytes 8 位优化器和梯度检查点来帮助训练 DreamBooth 模型。安装 bitsandbytes：

```py
pip install bitsandbytes
```

然后，将以下参数添加到您的训练命令中：

```bash
accelerate launch train_dreambooth.py \
  --gradient_checkpointing \
  --use_8bit_adam \
```

</hfoption>
<hfoption id="12GB">

在 12GB GPU 上，您需要 bitsandbytes 8 位优化器、梯度检查点、xFormers，并将梯度设置为 `None` 而不是零以减少内存使用。

```bash
accelerate launch train_dreambooth.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
```

</hfoption>
<hfoption id="8GB">

在 8GB GPU 上，您需要 [DeepSpeed](https://www.deepspeed.ai/) 将一些张量从 vRAM 卸载到 CPU 或 NVME，以便在更少的 GPU 内存下进行训练。

运行以下命令来配置您的 🤗 Accelerate 环境：

```bash
accelerate config
```

在配置过程中，确认您想使用 DeepSpeed。现在，通过结合 DeepSpeed 阶段 2、fp16 混合精度以及将模型参数和优化器状态卸载到 CPU，应该可以在低于 8GB vRAM 的情况下进行训练。缺点是这需要更多的系统 RAM（约 25 GB）。有关更多配置选项，请参阅 [DeepSpeed 文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)。

您还应将默认的 Adam 优化器更改为 DeepSpeed 的优化版本 [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) 以获得显著的速度提升。启用 `DeepSpeedCPUAdam` 要求您的系统 CUDA 工具链版本与 PyTorch 安装的版本相同。

目前，bitsandbytes 8 位优化器似乎与 DeepSpeed 不兼容。

就是这样！您不需要向训练命令添加任何额外参数。

</hfoption>
</hfoptions>

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_
saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
<hfoption id="Flax">

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400 \
  --push_to_hub
```

</hfoption>
</hfoptions>

训练完成后，您可以使用新训练的模型进行推理！

> [!TIP]
> 等不及在训练完成前就尝试您的模型进行推理？🤭 请确保安装了最新版本的 🤗 Accelerate。
>
> ```py
> from diffusers import DiffusionPipeline, UNet2DConditionModel
> from transformers import CLIPTextModel
> import torch
>
> unet = UNet2DConditionModel.from_pretrained("path/to/model/checkpoint-100/unet")
>
> # 如果您使用了 `--args.train_text_encoder` 进行训练，请确保也加载文本编码器
> text_encoder = CLIPTextModel.from_pretrained("path/to/model/checkpoint-100/checkpoint-100/text_encoder")
>
> pipeline = DiffusionPipeline.from_pretrained(
>     "stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16,
> ).to("cuda")
>
> image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
> image.save("dog-bucket.png")
> ```

<hfoptions id="training-inference">
<hfoption id="PyTorch">

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```

</hfoption>
<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path-to-your-trained-model", dtype=jax.numpy.bfloat16)

prompt = "A photo of sks dog in a bucket"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# 分片输入和随机数生成器
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_
steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("dog-bucket.png")
```

</hfoption>
</hfoptions>

## LoRA

LoRA 是一种训练技术，可显著减少可训练参数的数量。因此，训练速度更快，并且更容易存储生成的权重，因为它们小得多（约 100MB）。使用 [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) 脚本通过 LoRA 进行训练。

LoRA 训练脚本在 [LoRA 训练](lora) 指南中有更详细的讨论。

## Stable Diffusion XL

Stable Diffusion XL (SDXL) 是一个强大的文本到图像模型，可生成高分辨率图像，并在其架构中添加了第二个文本编码器。使用 [train_dreambooth_lora_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) 脚本通过 LoRA 训练 SDXL 模型。

SDXL 训练脚本在 [SDXL 训练](sdxl) 指南中有更详细的讨论。

## DeepFloyd IF

DeepFloyd IF 是一个级联像素扩散模型，包含三个阶段。第一阶段生成基础图像，第二和第三阶段逐步将基础图像放大为高分辨率 1024x1024 图像。使用 [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py) 或 [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) 脚本通过 LoRA 或完整模型训练 DeepFloyd IF 模型。

DeepFloyd IF 使用预测方差，但 Diffusers 训练脚本使用预测误差，因此训练的 DeepFloyd IF 模型被切换到固定方差调度。训练脚本将为您更新完全训练模型的调度器配置。但是，当您加载保存的 LoRA 权重时，还必须更新管道的调度器配置。

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", use_safetensors=True)

pipe.load_lora_weights("<lora weights path>")

# 更新调度器配置为固定方差调度
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

第二阶段模型需要额外的验证图像进行放大。您可以下载并使用训练图像的缩小版本。

```py
from huggingface_hub import snapshot_download

local_dir = "./dog_downsized"
snapshot_download(
    "diffusers/dog-example-downsized",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

以下代码示例简要概述了如何结合 DreamBooth 和 LoRA 训练 DeepFloyd IF 模型。一些需要注意的重要参数包括：

* `--resolution=64`，需要更小的分辨率，因为 DeepFloyd IF 是
一个像素扩散模型，用于处理未压缩的像素，输入图像必须更小
* `--pre_compute_text_embeddings`，提前计算文本嵌入以节省内存，因为 [`~transformers.T5Model`] 可能占用大量内存
* `--tokenizer_max_length=77`，您可以使用更长的默认文本长度与 T5 作为文本编码器，但默认模型编码过程使用较短的文本长度
* `--text_encoder_use_attention_mask`，将注意力掩码传递给文本编码器

<hfoptions id="IF-DreamBooth">
<hfoption id="Stage 1 LoRA DreamBooth">

使用 LoRA 和 DreamBooth 训练 DeepFloyd IF 的第 1 阶段需要约 28GB 内存。

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_lora"

accelerate launch train_dreambooth_lora.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --scale_lr \
  --max_train_steps=1200 \
  --validation_prompt="a sks dog" \
  --validation_epochs=25 \
  --checkpointing_steps=100 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask
```

</hfoption>
<hfoption id="Stage 2 LoRA DreamBooth">

对于使用 LoRA 和 DreamBooth 的 DeepFloyd IF 第 2 阶段，请注意这些参数：

* `--validation_images`，验证期间用于上采样的图像
* `--class_labels_conditioning=timesteps`，根据需要额外条件化 UNet，如第 2 阶段中所需
* `--learning_rate=1e-6`，与第 1 阶段相比使用较低的学习率
* `--resolution=256`，上采样器的预期分辨率

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

python train_dreambooth_lora.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a sks dog" \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --max_train_steps=2000 \
    --validation_prompt="a sks dog" \
    --validation_epochs=100 \
    --checkpointing_steps=500 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --validation_images $VALIDATION_IMAGES \
    --class_labels_conditioning=timesteps
```

</hfoption>
<hfoption id="Stage 1 DreamBooth">

对于使用 DreamBooth 的 DeepFloyd IF 第 1 阶段，请注意这些参数：

* `--skip_save_text_encoder`，跳过保存完整 T5 文本编码器与微调模型
* `--use_8bit_adam`，使用 8 位 Adam 优化器以节省内存，因为
     
优化器状态的大小在训练完整模型时
* `--learning_rate=1e-7`，对于完整模型训练应使用非常低的学习率，否则模型质量会下降（您可以使用更高的学习率和更大的批次大小）

使用8位Adam和批次大小为4进行训练，完整模型可以在约48GB内存下训练。

```bash
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_if"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-7 \
  --max_train_steps=150 \
  --validation_prompt "a photo of sks dog" \
  --validation_steps 25 \
  --text_encoder_use_attention_mask \
  --tokenizer_max_length 77 \
  --pre_compute_text_embeddings \
  --use_8bit_adam \
  --set_grads_to_none \
  --skip_save_text_encoder \
  --push_to_hub
```

</hfoption>
<hfoption id="Stage 2 DreamBooth">

对于DeepFloyd IF的第二阶段DreamBooth，请注意这些参数：

* `--learning_rate=5e-6`，使用较低的学习率和较小的有效批次大小
* `--resolution=256`，上采样器的预期分辨率
* `--train_batch_size=2` 和 `--gradient_accumulation_steps=6`，为了有效训练包含面部的图像，需要更大的批次大小

```bash
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

accelerate launch train_dreambooth.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=256 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=6 \
  --learning_rate=5e-6 \
  --max_train_steps=2000 \
  --validation_prompt="a sks dog" \
  --validation_steps=150 \
  --checkpointing_steps=500 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask \
  --validation_images $VALIDATION_IMAGES \
  --class_labels_conditioning timesteps \
  --push_to_hub
```

</hfoption>
</hfoptions>

### 训练技巧

训练DeepFloyd IF模型可能具有挑战性，但以下是我们发现有用的技巧：

- LoRA对于训练第一阶段模型已足够，因为模型的低分辨率使得表示更精细的细节变得困难，无论如何。
- 对于常见或简单的对象，您不一定需要微调上采样器。确保传递给上采样器的提示被调整以移除实例提示中的新令牌。例如，如果您第一阶段提示是"a sks dog"，那么您第二阶段的提示应该是"a dog"。
- 对于更精细的细节，如面部，完全训练
使用阶段2上采样器比使用LoRA训练阶段2模型更好。使用更大的批次大小和较低的学习率也有帮助。
- 应使用较低的学习率来训练阶段2模型。
- [`DDPMScheduler`] 比训练脚本中使用的DPMSolver效果更好。

## 下一步

恭喜您训练了您的DreamBooth模型！要了解更多关于如何使用您的新模型的信息，以下指南可能有所帮助：
- 如果您使用LoRA训练了您的模型，请学习如何[加载DreamBooth](../using-diffusers/loading_adapters)模型进行推理。