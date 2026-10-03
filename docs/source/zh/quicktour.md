<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# 快速上手

训练扩散模型，是为了对随机高斯噪声进行逐步去噪，以生成令人感兴趣的样本，比如图像或者语音。

扩散模型的发展引起了人们对生成式人工智能的极大兴趣，你可能已经在网上见过扩散生成的图像了。🧨 Diffusers库的目的是让大家更易上手扩散模型。

无论你是开发人员还是普通用户，本文将向你介绍🧨 Diffusers 并帮助你快速开始生成内容！

🧨 Diffusers 库的三个主要组件：


无论你是开发者还是普通用户，这个快速指南将向你介绍🧨 Diffusers，并帮助你快速使用和生成！该库三个主要部分如下：

* [`DiffusionPipeline`]是一个高级的端到端类，旨在通过预训练的扩散模型快速生成样本进行推理。
* 作为创建扩散系统做组件的流行的预训练[模型](./api/models)框架和模块。
* 许多不同的[调度器](./api/schedulers/overview)：控制如何在训练过程中添加噪声的算法，以及如何在推理过程中生成去噪图像的算法。

快速入门将告诉你如何使用[`DiffusionPipeline`]进行推理，然后指导你如何结合模型和调度器以复现[`DiffusionPipeline`]内部发生的事情。

> [!TIP]
> 快速入门是🧨[Diffusers入门](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)的简化版，可以帮助你快速上手。如果你想了解更多关于🧨 Diffusers的目标、设计理念以及关于它的核心API的更多细节，可以点击🧨[Diffusers入门](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)查看。

在开始之前，确认一下你已经安装好了所需要的库：

```bash
pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) 在推理和训练过程中加速模型加载。
- [🤗 Transformers](https://huggingface.co/docs/transformers/index) 是运行最流行的扩散模型所必须的库，比如[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview).

## 扩散模型管道

[`DiffusionPipeline`]是用预训练的扩散系统进行推理的最简单方法。它是一个包含模型和调度器的端到端系统。你可以直接使用[`DiffusionPipeline`]完成许多任务。请查看下面的表格以了解一些支持的任务，要获取完整的支持任务列表，请查看[🧨 Diffusers 总结](./api/pipelines/overview#diffusers-summary) 。

| **任务**                     | **描述**                                                                                              | **管道**
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation          | 从高斯噪声中生成图片 | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | 给定文本提示生成图像 | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation     | 在文本提示的指导下调整图像 | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting          | 给出图像、遮罩和文本提示，填充图像的遮罩部分 | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | 在文本提示的指导下调整图像的部分内容，同时通过深度估计保留其结构 | [depth2img](./using-diffusers/depth2img) |

首先创建一个[`DiffusionPipeline`]的实例，并指定要下载的pipeline检查点。
你可以使用存储在Hugging Face Hub上的任何[`DiffusionPipeline`][检查点](https://huggingface.co/models?library=diffusers&sort=downloads)。
在教程中，你将加载[`stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)检查点，用于文本到图像的生成。

首先创建一个[DiffusionPipeline]实例，并指定要下载的管道检查点。
您可以在Hugging Face Hub上使用[DiffusionPipeline]的任何检查点。
在本快速入门中，您将加载stable-diffusion-v1-5检查点，用于文本到图像生成。

> [!WARNING]
> 。
>
> 对于[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)模型，在运行该模型之前，请先仔细阅读[许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。🧨 Diffusers实现了一个[`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)，以防止有攻击性的或有害的内容，但Stable Diffusion模型改进图像的生成能力仍有可能产生潜在的有害内容。

用[`~DiffusionPipeline.from_pretrained`]方法加载模型。

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
```
[`DiffusionPipeline`]会下载并缓存所有的建模、标记化和调度组件。你可以看到Stable Diffusion的pipeline是由[`UNet2DConditionModel`]和[`PNDMScheduler`]等组件组成的：

```py
>>> pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.13.1",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

我们强烈建议你在GPU上运行这个pipeline，因为该模型由大约14亿个参数组成。

你可以像在Pytorch里那样把生成器对象移到GPU上：

```python
>>> pipeline.to("cuda")
```

现在你可以向`pipeline`传递一个文本提示来生成图像，然后获得去噪的图像。默认情况下，图像输出被放在一个[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)对象中。

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>


调用`save`保存图像:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### 本地管道

你也可以在本地使用管道。唯一的区别是你需提前下载权重：

```
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

将下载好的权重加载到管道中:

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
```

现在你可以像上一节中那样运行管道了。

### 更换调度器

不同的调度器对去噪速度和质量的权衡是不同的。要想知道哪种调度器最适合你，最好的办法就是试用一下。🧨 Diffusers的主要特点之一是允许你轻松切换不同的调度器。例如，要用[`EulerDiscreteScheduler`]替换默认的[`PNDMScheduler`]，用[`~diffusers.ConfigMixin.from_config`]方法加载即可：

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```


试着用新的调度器生成一个图像，看看你能否发现不同之处。

在下一节中，你将仔细观察组成[`DiffusionPipeline`]的组件——模型和调度器，并学习如何使用这些组件来生成猫咪的图像。

## 模型

大多数模型取一个噪声样本，在每个时间点预测*噪声残差*（其他模型则直接学习预测前一个样本或速度或[`v-prediction`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110)），即噪声较小的图像与输入图像的差异。你可以混搭模型创建其他扩散系统。

模型是用[`~ModelMixin.from_pretrained`]方法启动的，该方法还在本地缓存了模型权重，所以下次加载模型时更快。对于快速入门，你默认加载的是[`UNet2DModel`]，这是一个基础的无条件图像生成模型，该模型有一个在猫咪图像上训练的检查点：


```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id)
```

想知道模型的参数，调用 `model.config`:

```py
>>> model.config
```

模型配置是一个🧊冻结的🧊字典，意思是这些参数在模型创建后就不变了。这是特意设置的，确保在开始时用于定义模型架构的参数保持不变，其他参数仍然可以在推理过程中进行调整。

一些最重要的参数：

* `sample_size`：输入样本的高度和宽度尺寸。
* `in_channels`：输入样本的输入通道数。
* `down_block_types`和`up_block_types`：用于创建U-Net架构的下采样和上采样块的类型。
* `block_out_channels`：下采样块的输出通道数；也以相反的顺序用于上采样块的输入通道数。
* `layers_per_block`：每个U-Net块中存在的ResNet块的数量。

为了使用该模型进行推理，用随机高斯噪声生成图像形状。它应该有一个`batch`轴，因为模型可以接收多个随机噪声，一个`channel`轴，对应于输入通道的数量，以及一个`sample_size`轴，对应图像的高度和宽度。


```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

对于推理，将噪声图像和一个`timestep`传递给模型。`timestep` 表示输入图像的噪声程度，开始时噪声更多，结束时噪声更少。这有助于模型确定其在扩散过程中的位置，是更接近开始还是结束。使用 `sample` 获得模型输出：


```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

想生成实际的样本，你需要一个调度器指导去噪过程。在下一节中，你将学习如何把模型与调度器结合起来。

## 调度器

调度器管理一个噪声样本到一个噪声较小的样本的处理过程，给出模型输出 —— 在这种情况下，它是`noisy_residual`。



> [!TIP]
> 🧨 Diffusers是一个用于构建扩散系统的工具箱。预定义好的扩散系统[`DiffusionPipeline`]能方便你快速试用，你也可以单独选择自己的模型和调度器组件来建立一个自定义的扩散系统。

在快速入门教程中，你将用它的[`~diffusers.ConfigMixin.from_config`]方法实例化[`DDPMScheduler`]：

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_config(repo_id)
>>> scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.13.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

> [!TIP]
> 💡 注意调度器是如何从配置中实例化的。与模型不同，调度器没有可训练的权重，而且是无参数的。

* `num_train_timesteps`：去噪过程的长度，或者换句话说，将随机高斯噪声处理成数据样本所需的时间步数。
* `beta_schedule`：用于推理和训练的噪声表。
* `beta_start`和`beta_end`：噪声表的开始和结束噪声值。

要预测一个噪音稍小的图像，请将 模型输出、`timestep`和当前`sample` 传递给调度器的[`~diffusers.DDPMScheduler.step`]方法：


```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
```

这个 `less_noisy_sample` 去噪样本 可以被传递到下一个`timestep` ，处理后会将变得噪声更小。现在让我们把所有步骤合起来，可视化整个去噪过程。

首先，创建一个函数，对去噪后的图像进行后处理并显示为`PIL.Image`：

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

将输入和模型移到GPU上加速去噪过程：

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

现在创建一个去噪循环，该循环预测噪声较少样本的残差，并使用调度程序计算噪声较少的样本：

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

看！这样就从噪声中生成出一只猫了！😻

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## 下一步

希望你在这次快速入门教程中用🧨Diffuser 生成了一些很酷的图像! 下一步你可以:

* 在[训练](./tutorials/basic_training)教程中训练或微调一个模型来生成你自己的图像。
* 查看官方和社区的[训练或微调脚本](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples)的例子，了解更多使用情况。
* 在[使用不同的调度器](./using-diffusers/schedulers)指南中了解更多关于加载、访问、更改和比较调度器的信息。
* 在[Stable Diffusion](./stable_diffusion)教程中探索提示工程、速度和内存优化，以及生成更高质量图像的技巧。
* 通过[在GPU上优化PyTorch](./optimization/fp16)指南，以及运行[Apple (M1/M2)上的Stable Diffusion](./optimization/mps)和[ONNX Runtime](./optimization/onnx)的教程，更深入地了解如何加速🧨Diffuser。