# 将模型适配至新任务

许多扩散系统共享相同的组件架构，这使得您能够将针对某一任务预训练的模型调整适配至完全不同的新任务。

本指南将展示如何通过初始化并修改预训练 [`UNet2DConditionModel`] 的架构，将文生图预训练模型改造为图像修复(inpainting)模型。

## 配置 UNet2DConditionModel 参数

默认情况下，[`UNet2DConditionModel`] 的[输入样本](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel.in_channels)接受4个通道。例如加载 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 这样的文生图预训练模型，查看其 `in_channels` 参数值：

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
pipeline.unet.config["in_channels"]
4
```

而图像修复任务需要输入样本具有9个通道。您可以在 [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) 这样的预训练修复模型中验证此参数：

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting", use_safetensors=True)
pipeline.unet.config["in_channels"]
9
```

要将文生图模型改造为修复模型，您需要将 `in_channels` 参数从4调整为9。

初始化一个加载了文生图预训练权重的 [`UNet2DConditionModel`]，并将 `in_channels` 设为9。由于输入通道数变化导致张量形状改变，需要设置 `ignore_mismatched_sizes=True` 和 `low_cpu_mem_usage=False` 来避免尺寸不匹配错误。

```python
from diffusers import AutoModel

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet = AutoModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=9,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
)
```

此时文生图模型的其他组件权重仍保持预训练状态，但UNet的输入卷积层权重(`conv_in.weight`)会随机初始化。由于这一关键变化，必须对模型进行修复任务的微调，否则模型将仅会输出噪声。
