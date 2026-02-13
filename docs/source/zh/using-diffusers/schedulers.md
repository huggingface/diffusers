<!--Copyright 2025 The HuggingFace Team. All rights reserved.

根据 Apache License 2.0 许可证（以下简称"许可证"）授权；
除非符合许可证要求，否则不得使用本文件。
您可以通过以下链接获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，
无任何明示或暗示的担保或条件。详见许可证中关于权限和限制的具体规定。
-->

# 加载调度器与模型

[[open-in-colab]]

Diffusion管道是由可互换的调度器(schedulers)和模型(models)组成的集合，可通过混合搭配来定制特定用例的流程。调度器封装了整个去噪过程（如去噪步数和寻找去噪样本的算法），其本身不包含可训练参数，因此内存占用极低。模型则主要负责从含噪输入到较纯净样本的前向传播过程。

本指南将展示如何加载调度器和模型来自定义流程。我们将全程使用[stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5)检查点，首先加载基础管道：

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
```

通过`pipeline.scheduler`属性可查看当前管道使用的调度器：

```python
pipeline.scheduler
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.21.4",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}
```

## 加载调度器

调度器通过配置文件定义，同一配置文件可被多种调度器共享。使用[`SchedulerMixin.from_pretrained`]方法加载时，需指定`subfolder`参数以定位配置文件在仓库中的正确子目录。

例如加载[`DDIMScheduler`]：

```python
from diffusers import DDIMScheduler, DiffusionPipeline

ddim = DDIMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
```

然后将新调度器传入管道：

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler=ddim, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
```

## 调度器对比

不同调度器各有优劣，难以定量评估哪个最适合您的流程。通常需要在去噪速度与质量之间权衡。我们建议尝试多种调度器以找到最佳方案。通过`pipeline.scheduler.compatibles`属性可查看兼容当前管道的所有调度器。

下面我们使用相同提示词和随机种子，对比[`LMSDiscreteScheduler`]、[`EulerDiscreteScheduler`]、[`EulerAncestralDiscreteScheduler`]和[`DPMSolverMultistepScheduler`]的表现：

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
generator = torch.Generator(device="cuda").manual_seed(8)
```

使用[`~ConfigMixin.from_config`]方法加载不同调度器的配置来切换管道调度器：

<hfoptions id="schedulers">
<hfoption id="LMSDiscreteScheduler">

[`LMSDiscreteScheduler`]通常能生成比默认调度器更高质量的图像。

```python
from diffusers import LMSDiscreteScheduler

pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="EulerDiscreteScheduler">

[`EulerDiscreteScheduler`]仅需30步即可生成高质量图像。

```python
from diffusers import EulerDiscreteScheduler

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="EulerAncestralDiscreteScheduler">

[`EulerAncestralDiscreteScheduler`]同样可在30步内生成高质量图像。

```python
from diffusers import EulerAncestralDiscreteScheduler

pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
<hfoption id="DPMSolverMultistepScheduler">

[`DPMSolverMultistepScheduler`]在速度与质量间取得平衡，仅需20步即可生成优质图像。

```python
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, generator=generator).images[0]
image
```

</hfoption>
</hfoptions>

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_lms.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">LMSDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_discrete.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerDiscreteScheduler</figcaption>
  </div>
</div>
<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_euler_ancestral.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerAncestralDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/diffusers_docs/astronaut_dpm.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">DPMSolverMultistepScheduler</figcaption>
  </div>
</div>

多数生成图像质量相近，实际选择需根据具体场景测试多种调度器进行比较。

### Flax调度器

对比Flax调度器时，需额外将调度器状态加载到模型参数中。例如将[`FlaxStableDiffusionPipeline`]的默认调度器切换为超高效的[`FlaxDPMSolverMultistepScheduler`]：

> [!警告]
> [`FlaxLMSDiscreteScheduler`]和[`FlaxDDPMScheduler`]目前暂不兼容[`FlaxStableDiffusionPipeline`]。

```python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler

scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="scheduler"
)
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    scheduler=scheduler,
    variant="bf16",
    dtype=jax.numpy.bfloat16,
)
params["scheduler"] = scheduler_state
```

利用Flax对TPU的兼容性实现并行图像生成。需为每个设备复制模型参数，并分配输入数据：

```python
# 每个并行设备生成1张图像（TPUv2-8/TPUv3-8支持8设备并行）
prompt = "一张宇航员在火星上骑马的高清照片，高分辨率，高画质。"
num_samples = jax.device_count()
prompt_ids = pipeline.prepare_inputs([prompt] * num_samples)

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 25

# 分配输入和随机种子
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

## 模型加载

通过[`ModelMixin.from_pretrained`]方法加载模型，该方法会下载并缓存模型权重和配置的最新版本。若本地缓存已存在最新文件，则直接复用缓存而非重复下载。

通过`subfolder`参数可从子目录加载模型。例如[stable-diffusion-v1-5/stable-diffusion-v1-5](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5)的模型权重存储在[unet](https://hf.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet)子目录中：

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True)
```

也可直接从[仓库](https://huggingface.co/google/ddpm-cifar10-32/tree/main)加载：

```python
from diffusers import UNet2DModel

unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
```

加载和保存模型变体时，需在[`ModelMixin.from_pretrained`]和[`ModelMixin.save_pretrained`]中指定`variant`参数：

```python
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="non_ema", use_safetensors=True
)
unet.save_pretrained("./local-unet", variant="non_ema")
```

使用[`~ModelMixin.from_pretrained`]的`torch_dtype`参数指定模型加载精度：

```python
from diffusers import AutoModel

unet = AutoModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16
)
```

也可使用[torch.Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html)方法即时转换精度，但会转换所有权重（不同于`torch_dtype`参数会保留`_keep_in_fp32_modules`中的层）。这对某些必须保持fp32精度的层尤为重要（参见[示例](https://github.com/huggingface/diffusers/blob/f864a9a352fa4a220d860bfdd1782e3e5af96382/src/diffusers/models/transformers/transformer_wan.py#L374)）。
