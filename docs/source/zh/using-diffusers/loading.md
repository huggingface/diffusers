# 加载流程管道

[[在Colab中打开]]

扩散系统由参数化模型和调度器等多个组件组成，这些组件以复杂的方式交互。为此我们设计了[`DiffusionPipeline`]，将整个扩散系统的复杂性封装成简单易用的API。同时[`DiffusionPipeline`]完全可定制，您可以修改每个组件来构建适合自己需求的扩散系统。

本指南将展示如何加载：
- 从Hub和本地加载流程管道
- 将不同组件加载到流程管道中
- 在不增加内存使用的情况下加载多个流程管道
- 检查点变体，如不同的浮点类型或非指数移动平均(EMA)权重

## 加载流程管道

> [!TIP]
> 如果您对[`DiffusionPipeline`]类的工作原理感兴趣，可直接跳转到[DiffusionPipeline详解](#diffusionpipeline-explained)部分。

加载任务流程管道有两种方式：
1. 加载通用的[`DiffusionPipeline`]类，让它自动从检查点检测正确的流程管道类
2. 为特定任务加载特定的流程管道类

<hfoptions id="pipelines">
<hfoption id="通用流程管道">

[`DiffusionPipeline`]类是从[Hub](https://huggingface.co/models?library=diffusers&sort=trending)加载最新热门扩散模型的简单通用方法。它使用[`~DiffusionPipeline.from_pretrained`]方法自动从检查点检测任务的正确流程管道类，下载并缓存所有需要的配置和权重文件，返回一个可用于推理的流程管道。

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
```

同样的检查点也可以用于图像到图像任务。[`DiffusionPipeline`]类可以处理任何任务，只要您提供适当的输入。例如，对于图像到图像任务，您需要向流程管道传递初始图像。

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "丛林中的宇航员，彩色"
```
以下是您提供的英文内容的中文翻译，保持Diffusers、stable_diffusion、consisid等专有名词不译，并保留Markdown格式：

---

### 特定任务管道

若已知模型对应的具体管道类，可直接通过该类加载检查点。例如加载Stable Diffusion模型时，使用[`StableDiffusionPipeline`]类：

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
```

同一检查点也可用于其他任务（如图像到图像生成）。此时需改用对应任务的管道类，例如使用[`StableDiffusionImg2ImgPipeline`]加载相同检查点：

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
```

---

### 组件级数据类型指定

通过向`torch_dtype`参数传递字典，可为不同子模型定制数据类型。例如以`torch.bfloat16`精度加载transformer组件，其他组件默认使用`torch.float16`：

```python
from diffusers import HunyuanVideoPipeline
import torch

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype={"transformer": torch.bfloat16, "default": torch.float16},
)
print(pipe.transformer.dtype, pipe.vae.dtype)  # 输出: (torch.bfloat16, torch.float16)
```

若组件未在字典中显式指定且未设置`default`，将默认加载为`torch.float32`。

---

### 本地管道加载

使用[git-lfs](https://git-lfs.github.com/)手动下载检查点到本地后加载：

```bash
git-lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

下载完成后，将本地路径（如`./stable-diffusion-v1-5`）传递给[`~DiffusionPipeline.from_pretrained`]：

```python
from diffusers import DiffusionPipeline

stable_diffusion = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

---

### 内存需求评估工具

在下载前，可通过下方空间评估管道内存需求以确认硬件兼容性：

<div class="block dark:hidden">
	<iframe
        src="https://diffusers-compute-pipeline-size.hf.space?__theme=light"
        width="850"
        height="1600"
    ></iframe>
</div>
<div class="hidden dark:block">
    <iframe
        src="https://diffusers-compute-pipeline-size.hf.space?__theme=dark"
        width="850"
        height="1600"
    ></iframe>
</div>
### 自定义管道

您可以通过向管道中加载不同的组件来实现定制化。这一点非常重要，因为您可以：

- 根据需求切换为生成速度更快或生成质量更高的调度器（通过调用管道上的`scheduler.compatibles`方法查看兼容的调度器）
- 将默认管道组件替换为更新且性能更优的版本

例如，让我们用以下组件定制默认的[stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0)模型：

- 使用[`HeunDiscreteScheduler`]以牺牲生成速度为代价来生成更高质量的图像。必须传入`subfolder="scheduler"`参数到[`~HeunDiscreteScheduler.from_pretrained`]，以便从管道仓库的正确[子文件夹](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/scheduler)加载调度器配置。
- 使用在fp16模式下运行更稳定的VAE。

```python
from diffusers import StableDiffusionXLPipeline, HeunDiscreteScheduler, AutoencoderKL
import torch

scheduler = HeunDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
```

现在将新的调度器和VAE传入[`StableDiffusionXLPipeline`]：

```python
pipeline = StableDiffusionXLPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  scheduler=scheduler,
  vae=vae,
  torch_dtype=torch.float16,
  variant="fp16",
  use_safetensors=True
).to("cuda")
```

### 复用管道

当您加载多个共享相同模型组件的管道时，复用这些共享组件比重新将所有内容加载到内存中更有意义，尤其是在硬件内存受限的情况下。例如：

1. 您使用[`StableDiffusionPipeline`]生成了一张图像，但想通过[`StableDiffusionSAGPipeline`]提升其质量。这两个管道共享相同的预训练模型，因此加载两次会浪费内存。
2. 您想向从现有[`StableDiffusionPipeline`]实例化的[`AnimateDiffPipeline`]中添加一个模型组件（如[`MotionAdapter`](../api/pipelines/animatediff#animatediffpipeline)）。同样，这两个管道共享相同的预训练模型，加载全新管道会浪费内存。

通过[`DiffusionPipeline.from_pipe`] API，您可以在多个管道之间切换，利用它们的不同特性而不会增加内存使用。这类似于在管道中开启或关闭某个功能。

> [!提示]
> 若要在不同任务（而非功能）之间切换，请使用[`~DiffusionPipeline`]方法。
以下是您提供的英文内容的中文翻译，保持Diffusers、stable_diffusion、consisid等专有名词不变，并保留Markdown格式：

（此为文档10部分中的第4部分）

使用[`AutoPipeline`](../api/pipelines/auto_pipeline)类的[`~DiffusionPipeline.from_pipe`]方法可以自动根据任务识别管道类别（更多细节请参阅[AutoPipeline教程](../tutorials/autopipeline)）。

让我们从[`StableDiffusionPipeline`]开始，然后复用已加载的模型组件创建[`StableDiffusionSAGPipeline`]来提升生成质量。您将使用搭载[IP-Adapter](./ip_adapter)的[`StableDiffusionPipeline`]生成一张熊吃披萨的图片。

```python
from diffusers import DiffusionPipeline, StableDiffusionSAGPipeline
import torch
import gc
from diffusers.utils import load_image
from accelerate.utils import compute_module_sizes

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")

pipe_sd = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16)
pipe_sd.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_sd.set_ip_adapter_scale(0.6)
pipe_sd.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt="熊吃披萨",
    negative_prompt="白平衡错误, 昏暗, 草图, 最差质量, 低质量",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
).images[0]
out_sd
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sd_0.png"/>
</div>

作为参考，您可以查看此过程消耗的内存情况。

```python
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024
print(f"最大内存占用: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"最大内存占用: 4.406213283538818 GB"
```

现在通过[`~DiffusionPipeline.from_pipe`]方法，将[`StableDiffusionPipeline`]中的管道组件复用到[`StableDiffusionSAGPipeline`]中。

> [!警告]
> 某些管道方法在通过[`~DiffusionPipeline.from_pipe`]创建的新管道上可能无法正常工作。例如[`~DiffusionPipeline.enable_model_cpu_offload`]方法会根据每个管道独特的卸载序列在模型组件上安装钩子。如果新管道中模型执行顺序不同，CPU卸载可能无法正确工作。
>
> 为确保功能正常，我们建议对通过[`~DiffusionPipeline.from_pipe`]创建的新管道重新应用管道方法。

```python
pipe_sag = StableDiffusionSAGPipeline.from_pipe(
    pipe_sd
)

generator = torch.Generator(device="cpu").manual_seed(33)
out_sag = pipe_sag(
    prompt="熊吃披萨",
    negative_prompt="白平衡错误, 昏暗, 草图, 最差质量, 低质量",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
    guidance_scale=1.0,
    sag_scale=0.75
).images[0]
out_sag
```

<div class="flex justify-center">
  <!-- 图片将在此处显示 -->
</div>
<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sag_1.png"/>
</div>

如果检查内存使用情况，会发现与之前保持一致，因为 [`StableDiffusionPipeline`] 和 [`StableDiffusionSAGPipeline`] 共享相同的管道组件。这使得您可以互换使用它们，而无需额外的内存开销。

```py
print(f"最大内存分配量: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"最大内存分配量: 4.406213283538818 GB"
```

接下来，我们使用 [`AnimateDiffPipeline`] 为图像添加动画效果，同时向管道中添加 [`MotionAdapter`] 模块。对于 [`AnimateDiffPipeline`]，需要先卸载 IP-Adapter，并在创建新管道后重新加载（这一步骤仅适用于 [`AnimateDiffPipeline`]）。

```py
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

pipe_sag.unload_ip_adapter()
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

pipe_animate = AnimateDiffPipeline.from_pipe(pipe_sd, motion_adapter=adapter)
pipe_animate.scheduler = DDIMScheduler.from_config(pipe_animate.scheduler.config, beta_schedule="linear")
# 重新加载 IP-Adapter 和 LoRA 权重
pipe_animate.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_animate.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe_animate.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
pipe_animate.set_adapters("zoom-out", adapter_weights=0.75)
out = pipe_animate(
    prompt="熊吃披萨",
    num_frames=16,
    num_inference_steps=50,
    ip_adapter_image=image,
    generator=generator,
).frames[0]
export_to_gif(out, "out_animate.gif")
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_animate_3.gif"/>
</div>

[`AnimateDiffPipeline`] 对内存的需求更高，会消耗 15GB 内存（关于这对内存使用的影响，请参阅 [from_pipe 的内存使用情况](#memory-usage-of-from_pipe) 部分）。

```py
print(f"最大内存分配量: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
"最大内存分配量: 15.178664207458496 GB"
```

### 修改 from_pipe 组件

通过 [`~DiffusionPipeline.from_pipe`] 加载的管道可以使用不同的模型组件或方法进行自定义。但是，每当修改模型组件的状态时，会影响共享相同组件的所有其他管道。例如，如果在 [`StableDiffusionSAGPipeline`] 上调用 [`~diffusers.loaders.IPAdapterMixin.unload_ip_adapter`]，那么将无法在 [`StableDiffusionPipeline`] 中使用 IP-Adapter，因为它已从共享组件中移除。

```py
pipe.sag_unload_ip_adapter()

generator = to
```markdown
rch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt="熊吃披萨",
    negative_prompt="白平衡错误, 黑暗, 草图, 最差质量, 低质量",
    ip_adapter_image=image,
    num_inference_steps=50,
    generator=generator,
).images[0]
"AttributeError: 'NoneType' object has no attribute 'image_projection_layers'"
```

### from_pipe的内存占用

使用[`~DiffusionPipeline.from_pipe`]加载多个流程时，内存需求取决于内存占用最高的流程，与创建的流程数量无关。

| 流程类型 | 内存占用 (GB) |
|---|---|
| StableDiffusionPipeline | 4.400 |
| StableDiffusionSAGPipeline | 4.400 |
| AnimateDiffPipeline | 15.178 |

由于[`AnimateDiffPipeline`]内存需求最高，因此*总内存占用*仅基于该流程。只要后续创建的流程内存需求不超过[`AnimateDiffPipeline`]，内存占用就不会增加。各流程可交替使用，不会产生额外内存开销。

## 安全检测器

Diffusers为Stable Diffusion模型实现了[安全检测器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)，用于筛查可能生成的有害内容。该检测器会将生成输出与已知的NSFW内容硬编码库进行比对。如需禁用安全检测器，可向[`~DiffusionPipeline.from_pretrained`]方法传递`safety_checker=None`参数。

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, use_safetensors=True)
"""
您已通过传递`safety_checker=None`禁用了<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>的安全检测器。请确保遵守Stable Diffusion许可条款，避免在公开服务或应用中展示未过滤结果。Diffusers团队和Hugging Face强烈建议在所有面向公众的场景中保持安全过滤器启用，仅在进行网络行为分析或结果审计时禁用。更多信息请参阅https://github.com/huggingface/diffusers/pull/254。
"""
```

## 检查点变体

检查点变体通常指以下两种权重类型：

- 存储为不同浮点类型（如[torch.float16](https://pytorch.org/docs/stable/tensors.html#data-types)）的检查点，下载仅需一半带宽和存储空间。但继续训练或使用CPU时不可用此变体。
- 非指数移动平均（Non-EMA）权重，此类变体不应用于推理，仅适用于继续微调模型。

> [!提示]
> 当检查点包含...（后续内容待补充）
### 模型变体

即使模型结构完全相同，但如果它们在不同的数据集上训练或采用了不同的训练配置，就应当存放在独立的代码库中。例如，[stabilityai/stable-diffusion-2](https://hf.co/stabilityai/stable-diffusion-2) 和 [stabilityai/stable-diffusion-2-1](https://hf.co/stabilityai/stable-diffusion-2-1) 就分别存储在不同的代码库中。

反之，若某个变体与原检查点**完全一致**，则意味着它们具有相同的序列化格式（如 [safetensors](./using_safetensors)）、完全一致的模型结构，且所有张量权重形状均相同。

| **检查点类型**       | **权重文件名**                             | **加载参数**               |
|----------------------|-------------------------------------------|---------------------------|
| 原始版本             | diffusion_pytorch_model.safetensors        | -                         |
| 浮点精度变体         | diffusion_pytorch_model.fp16.safetensors   | `variant`, `torch_dtype`  |
| 非EMA变体            | diffusion_pytorch_model.non_ema.safetensors| `variant`                 |

加载变体时有两个关键参数：

- `torch_dtype` 指定加载权重的浮点精度。例如，若想通过加载 fp16 变体节省带宽，需同时设置 `variant="fp16"` 和 `torch_dtype=torch.float16` 以*将权重转换为 fp16*。若仅设置 `torch_dtype=torch.float16`，系统会先下载默认的 fp32 权重再执行转换。

- `variant` 指定从代码库加载哪个文件。例如，要从 [stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) 加载 UNet 的非EMA变体，需设置 `variant="non_ema"` 来下载对应的 `non_ema` 文件。

<hfoptions id="variants">
<hfoption id="fp16">

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    variant="fp16", 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
```

</hfoption>
<hfoption id="non-EMA">

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    variant="non_ema", 
    use_safetensors=True
)
```

</hfoption>
</hfoptions>

通过 [`DiffusionPipeline.save_pretrained`] 方法的 `variant` 参数，可将检查点保存为不同浮点精度或非EMA变体。建议将变体保存在原始检查点同一目录下，以便从同一位置加载不同版本。

<hfoptions id="save">
<hfoption id="fp16">

```python
from diffusers import DiffusionPipeline

pipeline.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16")
```

</hfoption>
<hfoption id="non_ema">

```python
pipeline.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="non_ema")
```

</hfoption>
</hfoptions>
以下是您提供的英文内容的中文翻译，保留了Diffusers、stable_diffusion、consisid等专有名词的英文形式，并维持了Markdown格式：

```markdown
on-v1-5/stable-diffusion-v1-5", variant="non_ema")
```

</hfoption>
</hfoptions>

如果不将变体保存到现有文件夹中，则必须指定 `variant` 参数，否则会抛出 `Exception` 异常，因为它无法找到原始检查点。

```python
# 👎 这种方式不可行
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# 👍 这种方式可行
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```

## DiffusionPipeline 原理解析

作为类方法，[`DiffusionPipeline.from_pretrained`] 主要承担两项职责：

- 下载推理所需的最新版文件夹结构并缓存。若本地缓存中已存在最新文件夹结构，[`DiffusionPipeline.from_pretrained`] 会直接复用缓存而不会重复下载文件。
- 将缓存的权重加载至正确的流水线[类](../api/pipelines/overview#diffusers-summary)（该信息从 `model_index.json` 文件中获取），并返回其实例。

流水线的底层文件夹结构与其类实例直接对应。例如，[`StableDiffusionPipeline`] 对应 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 中的文件夹结构。

```python
from diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```

您会看到 pipeline 是 [`StableDiffusionPipeline`] 的实例，由七个组件构成：

- `"feature_extractor"`：来自 🤗 Transformers 的 [`~transformers.CLIPImageProcessor`]。
- `"safety_checker"`：用于屏蔽有害内容的[组件](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32)。
- `"scheduler"`：[`PNDMScheduler`] 的实例。
- `"text_encoder"`：来自 🤗 Transformers 的 [`~transformers.CLIPTextModel`]。
- `"tokenizer"`：来自 🤗 Transformers 的 [`~transformers.CLIPTokenizer`]。
- `"unet"`：[`UNet2DConditionModel`] 的实例。
- `"vae"`：[`AutoencoderKL`] 的实例。

```json
StableDiffusionPipeline {
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
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

将流水线实例的组件与 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusio
```

（注：由于您提供的英文内容在结尾处被截断，中文翻译也保持相同截断位置。若需完整翻译最后部分，请提供剩余内容。）
以下是您提供的英文内容的中文翻译，已按要求保留Diffusers、stable_diffusion等专有名词不翻译，并保持Markdown格式：

（这是文档10部分中的第9部分）

观察[stable_diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)的文件夹结构，您会发现仓库中每个组件都有独立的文件夹：

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   |── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
|   |── diffusion_pytorch_model.fp16.bin
│   |── diffusion_pytorch_model.f16.safetensors
│   |── diffusion_pytorch_model.non_ema.bin
│   |── diffusion_pytorch_model.non_ema.safetensors
│   └── diffusion_pytorch_model.safetensors
|── vae
.   ├── config.json
.   ├── diffusion_pytorch_model.bin
    ├── diffusion_pytorch_model.fp16.bin
    ├── diffusion_pytorch_model.fp16.safetensors
    └── diffusion_pytorch_model.safetensors
```

您可以通过属性访问管道的每个组件来查看其配置：

```py
pipeline.tokenizer
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad_token": "<|endoftext|>",
    },
    clean_up_tokenization_spaces=True
)
```

每个管道都需要一个[`model_index.json`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json)文件，该文件会告诉[`DiffusionPipeline`]：

- 从`_class_name`加载哪个管道类
- 创建模型时使用的🧨 Diffusers版本`_diffusers_version`
- 子文件夹中存储了哪些库的哪些组件（`name`对应组件和子文件夹名称，`library`对应要加载类的库名，`class`对应类名）

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
```
```json
{
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
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
