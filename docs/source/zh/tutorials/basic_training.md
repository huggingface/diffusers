<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# 训练扩散模型

无条件图像生成是扩散模型最常见的应用之一，它会生成与训练数据集风格相似的图像。通常来说，在某个特定数据集上微调预训练模型能得到最好的结果。你可以在 [Hub](https://huggingface.co/search/full-text?q=unconditional-image-generation&type=model) 上找到很多现成检查点；如果找不到满意的，也完全可以自己训练一个！

这篇教程会教你如何在 [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) 数据集的一个子集上，从零开始训练一个 [`UNet2DModel`]，生成属于你自己的 🦋 蝴蝶图像 🦋。

> [!TIP]
> 💡 这篇训练教程基于 [Training with 🧨 Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook 编写。如果你想了解更多背景，例如扩散模型的工作原理，也推荐一起看看这个 notebook。

开始之前，请确认已经安装了 🤗 Datasets，用来加载和预处理图像数据集；以及 🤗 Accelerate，用来简化任意数量 GPU 上的训练。下面这条命令也会安装 [TensorBoard](https://www.tensorflow.org/tensorboard) 来可视化训练指标（你也可以使用 [Weights & Biases](https://docs.wandb.ai/) 跟踪训练）。

```py
# 如果你在 Colab 中运行，请取消注释来安装所需依赖
#!pip install diffusers[training]
```

我们也很鼓励你把模型分享给社区。为此，你需要登录自己的 Hugging Face 账号（如果还没有，可以在 [这里](https://hf.co/join) 创建）。你可以在 notebook 中登录，系统会提示你输入 token。请确保这个 token 具有写入权限。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

或者在终端里登录：

```bash
hf auth login
```

由于模型检查点通常比较大，建议安装 [Git-LFS](https://git-lfs.com/) 来管理这些大文件：

```bash
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```

## 训练配置

为了方便起见，我们先创建一个 `TrainingConfig` 类，把训练超参数放在一起（你可以按需调整）：

```py
>>> from dataclasses import dataclass

>>> @dataclass
... class TrainingConfig:
...     image_size = 128  # 生成图像的分辨率
...     train_batch_size = 16
...     eval_batch_size = 16  # 评估时每次采样多少张图像
...     num_epochs = 50
...     gradient_accumulation_steps = 1
...     learning_rate = 1e-4
...     lr_warmup_steps = 500
...     save_image_epochs = 10
...     save_model_epochs = 30
...     mixed_precision = "fp16"  # float32 用 `no`，自动混合精度用 `fp16`
...     output_dir = "ddpm-butterflies-128"  # 本地和 HF Hub 上的模型名称

...     push_to_hub = True  # 是否将保存后的模型上传到 HF Hub
...     hub_model_id = "<your-username>/<my-awesome-model>"  # 在 HF Hub 上创建的仓库名称
...     hub_private_repo = None
...     overwrite_output_dir = True  # 重新运行 notebook 时是否覆盖旧模型
...     seed = 0


>>> config = TrainingConfig()
```

## 加载数据集

你可以很轻松地通过 🤗 Datasets 加载 [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) 数据集：

```py
>>> from datasets import load_dataset

>>> config.dataset_name = "huggan/smithsonian_butterflies_subset"
>>> dataset = load_dataset(config.dataset_name, split="train")
```

> [!TIP]
> 💡 你也可以从 [HugGan Community Event](https://huggingface.co/huggan) 找到更多数据集，或者通过本地 [`ImageFolder`](https://huggingface.co/docs/datasets/image_dataset#imagefolder) 使用自己的数据集。如果你使用 HugGan Community Event 里的数据集，把 `config.dataset_name` 设为对应数据集的 repository id；如果你使用自己的图像，就设为 `imagefolder`。

🤗 Datasets 使用 [`~datasets.Image`] 特性自动解码图像数据，并将其加载为 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)，所以我们可以直接可视化：

```py
>>> import matplotlib.pyplot as plt

>>> fig, axs = plt.subplots(1, 4, figsize=(16, 4))
>>> for i, image in enumerate(dataset[:4]["image"]):
...     axs[i].imshow(image)
...     axs[i].set_axis_off()
>>> fig.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_ds.png"/>
</div>

不过这些图像的尺寸各不相同，所以你需要先做预处理：

* `Resize` 把图像缩放到 `config.image_size` 中定义的大小。
* `RandomHorizontalFlip` 通过随机水平翻转图像来做数据增强。
* `Normalize` 很重要，它会把像素值缩放到 `[-1, 1]` 区间，这是模型期望的输入范围。

```py
>>> from torchvision import transforms

>>> preprocess = transforms.Compose(
...     [
...         transforms.Resize((config.image_size, config.image_size)),
...         transforms.RandomHorizontalFlip(),
...         transforms.ToTensor(),
...         transforms.Normalize([0.5], [0.5]),
...     ]
... )
```

使用 🤗 Datasets 的 [`~datasets.Dataset.set_transform`] 方法，在训练过程中按需应用 `preprocess` 函数：

```py
>>> def transform(examples):
...     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
...     return {"images": images}


>>> dataset.set_transform(transform)
```

你也可以再次可视化图像，确认它们已经被调整到目标尺寸。接下来，就可以把数据集封装成一个 [DataLoader](https://pytorch.org/docs/stable/data#torch.utils.data.DataLoader) 来训练了！

```py
>>> import torch

>>> train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
```

## 创建 UNet2DModel

在 🧨 Diffusers 中，可以很方便地通过模型类和参数创建预训练模型。例如，下面创建一个 [`UNet2DModel`]：

```py
>>> from diffusers import UNet2DModel

>>> model = UNet2DModel(
...     sample_size=config.image_size,  # 目标图像分辨率
...     in_channels=3,  # 输入通道数，RGB 图像为 3
...     out_channels=3,  # 输出通道数
...     layers_per_block=2,  # 每个 UNet block 中使用多少个 ResNet 层
...     block_out_channels=(128, 128, 256, 256, 512, 512),  # 每个 UNet block 的输出通道数
...     down_block_types=(
...         "DownBlock2D",  # 标准的 ResNet 下采样块
...         "DownBlock2D",
...         "DownBlock2D",
...         "DownBlock2D",
...         "AttnDownBlock2D",  # 带空间自注意力的 ResNet 下采样块
...         "DownBlock2D",
...     ),
...     up_block_types=(
...         "UpBlock2D",  # 标准的 ResNet 上采样块
...         "AttnUpBlock2D",  # 带空间自注意力的 ResNet 上采样块
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...         "UpBlock2D",
...     ),
... )
```

通常最好先快速检查一下，样本图像的形状和模型输出形状是否一致：

```py
>>> sample_image = dataset[0]["images"].unsqueeze(0)
>>> print("Input shape:", sample_image.shape)
Input shape: torch.Size([1, 3, 128, 128])

>>> print("Output shape:", model(sample_image, timestep=0).sample.shape)
Output shape: torch.Size([1, 3, 128, 128])
```

很好！接下来，你还需要一个调度器为图像添加噪声。

## 创建调度器

调度器在训练和推理中的行为不同。推理时，调度器会从噪声中生成图像；训练时，调度器会取扩散过程某一步的模型输出或样本，并根据*噪声日程*与*更新规则*对图像加噪。

我们先看看 [`DDPMScheduler`]，并使用 `add_noise` 方法给前面的 `sample_image` 添加一些随机噪声：

```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import DDPMScheduler

>>> noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
>>> noise = torch.randn(sample_image.shape)
>>> timesteps = torch.LongTensor([50])
>>> noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

>>> Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/noisy_butterfly.png"/>
</div>

模型训练的目标，就是预测添加到图像中的噪声。当前步骤的损失可以这样计算：

```py
>>> import torch.nn.functional as F

>>> noise_pred = model(noisy_image, timesteps).sample
>>> loss = F.mse_loss(noise_pred, noise)
```

## 训练模型

到这里，启动训练所需的大部分组件都准备好了，剩下的就是把它们拼起来。

首先，你需要一个优化器和一个学习率调度器：

```py
>>> from diffusers.optimization import get_cosine_schedule_with_warmup

>>> optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
>>> lr_scheduler = get_cosine_schedule_with_warmup(
...     optimizer=optimizer,
...     num_warmup_steps=config.lr_warmup_steps,
...     num_training_steps=(len(train_dataloader) * config.num_epochs),
... )
```

接着，你还需要一种评估模型的方法。评估时，我们可以使用 [`DDPMPipeline`] 生成一批示例图像，并把它们保存成一个网格图：

```py
>>> from diffusers import DDPMPipeline
>>> from diffusers.utils import make_image_grid
>>> import os

>>> def evaluate(config, epoch, pipeline):
...     # 从随机噪声采样图像（这就是反向扩散过程）
...     # 管道默认输出类型是 `List[PIL.Image]`
...     images = pipeline(
...         batch_size=config.eval_batch_size,
...         generator=torch.Generator(device='cpu').manual_seed(config.seed), # 单独使用一个 torch generator，避免回退主训练循环的随机状态
...     ).images

...     # 把图像拼成网格
...     image_grid = make_image_grid(images, rows=4, cols=4)

...     # 保存图像
...     test_dir = os.path.join(config.output_dir, "samples")
...     os.makedirs(test_dir, exist_ok=True)
...     image_grid.save(f"{test_dir}/{epoch:04d}.png")
```

现在，你可以用 🤗 Accelerate 把这些组件包装进一个训练循环中，轻松实现 TensorBoard 日志记录、梯度累积和混合精度训练。为了把模型上传到 Hub，还需要写一个函数来创建仓库并将训练结果推送到 Hub。

> [!TIP]
> 💡 下面的训练循环看起来可能有点长，也有点吓人，但等你真正只用一行代码启动训练时，就会觉得很值得！如果你现在只想快点开始生成图像，也可以先直接复制运行下面的代码，之后再回头仔细研究训练循环，比如等模型训练完成的时候。🤗

```py
>>> from accelerate import Accelerator
>>> from huggingface_hub import create_repo, upload_folder
>>> from tqdm.auto import tqdm
>>> from pathlib import Path
>>> import os

>>> def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
...     # 初始化 accelerator 和 tensorboard 日志
...     accelerator = Accelerator(
...         mixed_precision=config.mixed_precision,
...         gradient_accumulation_steps=config.gradient_accumulation_steps,
...         log_with="tensorboard",
...         project_dir=os.path.join(config.output_dir, "logs"),
...     )
...     if accelerator.is_main_process:
...         if config.output_dir is not None:
...             os.makedirs(config.output_dir, exist_ok=True)
...         if config.push_to_hub:
...             repo_id = create_repo(
...                 repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
...             ).repo_id
...         accelerator.init_trackers("train_example")

...     # 准备所有对象
...     # 不需要记住固定顺序，只要解包时和传给 prepare 的顺序一致即可。
...     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
...         model, optimizer, train_dataloader, lr_scheduler
...     )

...     global_step = 0

...     # 开始训练模型
...     for epoch in range(config.num_epochs):
...         progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
...         progress_bar.set_description(f"Epoch {epoch}")

...         for step, batch in enumerate(train_dataloader):
...             clean_images = batch["images"]
...             # 为图像采样噪声
...             noise = torch.randn(clean_images.shape, device=clean_images.device)
...             bs = clean_images.shape[0]

...             # 为每张图像随机采样一个时间步
...             timesteps = torch.randint(
...                 0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
...                 dtype=torch.int64
...             )

...             # 按照每个时间步对应的噪声强度给干净图像加噪
...             # （这就是前向扩散过程）
...             noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

...             with accelerator.accumulate(model):
...                 # 预测噪声残差
...                 noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
...                 loss = F.mse_loss(noise_pred, noise)
...                 accelerator.backward(loss)

...                 if accelerator.sync_gradients:
...                     accelerator.clip_grad_norm_(model.parameters(), 1.0)
...                 optimizer.step()
...                 lr_scheduler.step()
...                 optimizer.zero_grad()

...             progress_bar.update(1)
...             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
...             progress_bar.set_postfix(**logs)
...             accelerator.log(logs, step=global_step)
...             global_step += 1

...         # 每个 epoch 后可以选择用 evaluate() 采样一些演示图像，并保存模型
...         if accelerator.is_main_process:
...             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

...             if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
...                 evaluate(config, epoch, pipeline)

...             if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
...                 if config.push_to_hub:
...                     upload_folder(
...                         repo_id=repo_id,
...                         folder_path=config.output_dir,
...                         commit_message=f"Epoch {epoch}",
...                         ignore_patterns=["step_*", "epoch_*"],
...                     )
...                 else:
...                     pipeline.save_pretrained(config.output_dir)
```

呼，这段代码确实不少！不过现在你终于可以用 🤗 Accelerate 的 [`~accelerate.notebook_launcher`] 函数启动训练了。把训练循环函数、所有训练参数以及进程数（你可以改成自己可用 GPU 的数量）传进去即可：

```py
>>> from accelerate import notebook_launcher

>>> args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

>>> notebook_launcher(train_loop, args, num_processes=1)
```

训练完成后，来看看你的扩散模型最终生成的 🦋 蝴蝶图像 🦋 吧！

```py
>>> import glob

>>> sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
>>> Image.open(sample_images[-1])
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/butterflies_final.png"/>
</div>

## 下一步

无条件图像生成只是可训练任务中的一个例子。你可以继续访问 [🧨 Diffusers 训练示例](../training/overview) 页面，探索更多任务和训练技术。比如：

* [Textual Inversion](../training/text_inversion)：教会模型一个特定的视觉概念，并把它融入生成结果中。
* [DreamBooth](../training/dreambooth)：给定某个主体的若干输入图像，生成该主体的个性化图像。
* [引导](../training/text2image)：在你自己的数据集上微调 Stable Diffusion 模型。
* [引导](../training/lora)：使用 LoRA 这种更省内存的方法，更快地微调超大模型。
