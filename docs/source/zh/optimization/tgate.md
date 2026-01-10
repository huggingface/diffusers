# T-GATE

[T-GATE](https://github.com/HaozheLiu-ST/T-GATE/tree/main) 通过跳过交叉注意力计算一旦收敛，加速了 [Stable Diffusion](../api/pipelines/stable_diffusion/overview)、[PixArt](../api/pipelines/pixart) 和 [Latency Consistency Model](../api/pipelines/latent_consistency_models.md) 管道的推理。此方法不需要任何额外训练，可以将推理速度提高 10-50%。T-GATE 还与 [DeepCache](./deepcache) 等其他优化方法兼容。

开始之前，请确保安装 T-GATE。

```bash
pip install tgate
pip install -U torch diffusers transformers accelerate DeepCache
```

要使用 T-GATE 与管道，您需要使用其对应的加载器。

| 管道 | T-GATE 加载器 |
|---|---|
| PixArt | TgatePixArtLoader |
| Stable Diffusion XL | TgateSDXLLoader |
| Stable Diffusion XL + DeepCache | TgateSDXLDeepCacheLoader |
| Stable Diffusion | TgateSDLoader |
| Stable Diffusion + DeepCache | TgateSDDeepCacheLoader |

接下来，创建一个 `TgateLoader`，包含管道、门限步骤（停止计算交叉注意力的时间步）和推理步骤数。然后在管道上调用 `tgate` 方法，提供提示、门限步骤和推理步骤数。

让我们看看如何为几个不同的管道启用此功能。

<hfoptions id="pipelines">
<hfoption id="PixArt">

使用 T-GATE 加速 `PixArtAlphaPipeline`：

```py
import torch
from diffusers import PixArtAlphaPipeline
from tgate import TgatePixArtLoader

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)

gate_step = 8
inference_step = 25
pipe = TgatePixArtLoader(
       pipe,
       gate_step=gate_step,
       num_inference_steps=inference_step,
).to("cuda")

image = pipe.tgate(
       "An alpaca made of colorful building blocks, cyberpunk.",
       gate_step=gate_step,
       num_inference_steps=inference_step,
).images[0]
```
</hfoption>
<hfoption id="Stable Diffusion XL">

使用 T-GATE 加速 `StableDiffusionXLPipeline`：

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLLoader

pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

gate_step = 10
inference_step = 25
pipe = TgateSDXLLoader(
       pipe,
       gate_step=gate_step,
       num_inference_steps=inference_step,
).to("cuda")

image = pipe.tgate(
       "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
       gate_step=gate_step,
       num_inference_steps=inference_step
).images[0]
```
</hfoption>
<hfoption id="StableDiffusionXL with DeepCache">

使用 [DeepCache](https://github.co 加速 `StableDiffusionXLPipeline`
m/horseee/DeepCache) 和 T-GATE：

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLDeepCacheLoader

pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

gate_step = 10
inference_step = 25
pipe = TgateSDXLDeepCacheLoader(
       pipe,
       cache_interval=3,
       cache_branch_id=0,
).to("cuda")

image = pipe.tgate(
       "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
       gate_step=gate_step,
       num_inference_steps=inference_step
).images[0]
```
</hfoption>
<hfoption id="Latent Consistency Model">

使用 T-GATE 加速 `latent-consistency/lcm-sdxl`：

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLLoader

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

gate_step = 1
inference_step = 4
pipe = TgateSDXLLoader(
       pipe,
       gate_step=gate_step,
       num_inference_steps=inference_step,
       lcm=True
).to("cuda")

image = pipe.tgate(
       "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
       gate_step=gate_step,
       num_inference_steps=inference_step
).images[0]
```
</hfoption>
</hfoptions>

T-GATE 还支持 [`StableDiffusionPipeline`] 和 [PixArt-alpha/PixArt-LCM-XL-2-1024-MS](https://hf.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS)。

## 基准测试
| 模型                 | MACs     | 参数     | 延迟 | 零样本 10K-FID on MS-COCO |
|-----------------------|----------|-----------|---------|---------------------------|
| SD-1.5                | 16.938T  | 859.520M  | 7.032s  | 23.927                    |
| SD-1.5 w/ T-GATE       | 9.875T   | 815.557M  | 4.313s  | 20.789                    |
| SD-2.1                | 38.041T  | 865.785M  | 16.121s | 22.609                    |
| SD-2.1 w/ T-GATE       | 22.208T  | 815.433 M | 9.878s  | 19.940                    |
| SD-XL                 | 149.438T | 2.570B    | 53.187s | 24.628                    |
| SD-XL w/ T-GATE        | 84.438T  | 2.024B    | 27.932s | 22.738                    |
| Pixart-Alpha          | 107.031T | 611.350M  | 61.502s | 38.669                    |
| Pixart-Alpha w/ T-GATE | 65.318T  | 462.585M  | 37.867s | 35.825                    |
| DeepCache (SD-XL)     | 57.888T  | -         | 19.931s | 23.755                    |
| DeepCache 配合 T-GATE    | 43.868T  | -         | 14.666秒 | 23.999                    |
| LCM (SD-XL)           | 11.955T  | 2.570B    | 3.805秒  | 25.044                    |
| LCM 配合 T-GATE          | 11.171T  | 2.024B    | 3.533秒  | 25.028                    |
| LCM (Pixart-Alpha)    | 8.563T   | 611.350M  | 4.733秒  | 36.086                    |
| LCM 配合 T-GATE          | 7.623T   | 462.585M  | 4.543秒  | 37.048                    |

延迟测试基于 NVIDIA 1080TI，MACs 和 Params 使用 [calflops](https://github.com/MrYxJ/calculate-flops.pytorch) 计算，FID 使用 [PytorchFID](https://github.com/mseitzer/pytorch-fid) 计算。