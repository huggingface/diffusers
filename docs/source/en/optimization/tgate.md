# T-GATE

[T-GATE](https://github.com/HaozheLiu-ST/T-GATE/tree/main) accelerates inference for [Stable Diffusion](../api/pipelines/stable_diffusion/overview), [PixArt](../api/pipelines/pixart), and [Latency Consistency Model](../api/pipelines/latent_consistency_models.md) pipelines by skipping the cross-attention calculation once it converges. This method doesn't require any additional training and it can speed up inference from 10-50%. T-GATE is also compatible with other optimization methods like [DeepCache](./deepcache).

Before you begin, make sure you install T-GATE.

```bash
pip install tgate
pip install -U pytorch diffusers transformers accelerate DeepCache


To use T-GATE with a pipeline, you need to use its corresponding loader.

| Pipeline | T-GATE Loader |
|---|---|
| PixArt | TgatePixArtLoader |
| Stable Diffusion XL | TgateSDXLLoader |
| Stable Diffusion XL + DeepCache | TgateSDXLDeepCacheLoader |

Next, create a `TgateLoader` with a pipeline, the gate step(`add brief description here`), and the number of inference steps. Then call the `tgate` method on the pipeline with a prompt, gate step, and the number of inference steps.

Let's see how to enable this for several different pipelines.

```py
import torch
from diffusers import PixArtAlphaPipeline
from tgate import TgatePixArtLoader

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
pipe = TgatePixArtLoader(
       pipe,
       gate_step=8,
       num_inference_steps=25,
)
pipe = pipe.to("cuda")

image = pipe.tgate(
       "An alpaca made of colorful building blocks, cyberpunk.",
        gate_step=gate_step,
       num_inference_steps=inference_step,
).images[0]

Accelerate `StableDiffusionXLPipeline` with TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
)

+ from tgate import TgateSDXLLoader
+ gate_step = 10
+ inference_step = 25
+ pipe = TgateSDXLLoader(
+        pipe,
+        gate_step=gate_step,
+        num_inference_steps=inference_step,
+ )

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         num_inference_steps=inference_step
+ ).images[0]
```

Accelerate `StableDiffusionXLPipeline` with [DeepCache](https://github.com/horseee/DeepCache) and TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
)

+ from tgate import TgateSDXLDeepCacheLoader
+ gate_step = 10
+ inference_step = 25
+ pipe = TgateSDXLDeepCacheLoader(
+        pipe,
+        cache_interval=3,
+        cache_branch_id=0,
+ )

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         num_inference_steps=inference_step
+ ).images[0]
```

Accelerate `latent-consistency/lcm-sdxl` with TGATE:

```diff
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler

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

+ from tgate import TgateSDXLLoader
+ gate_step = 1
+ inference_step = 4
+ pipe = TgateSDXLLoader(
+        pipe,
+        gate_step=gate_step,
+        num_inference_steps=inference_step,
+        lcm=True
+ )
pipe = pipe.to("cuda")

+ image = pipe.tgate(
+         "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
+         gate_step=gate_step,
+         num_inference_steps=inference_step
+ ).images[0]
```

TGATE also supports `StableDiffusionPipeline` and `PixArt-alpha/PixArt-LCM-XL-2-1024-MS`.
More details can be found at [here](https://github.com/HaozheLiu-ST/T-GATE/tree/release-v.0.1.0/main.py).

## 📄 Results
| Model                 | MACs     | Param     | Latency | Zero-shot 10K-FID on MS-COCO |
|-----------------------|----------|-----------|---------|---------------------------|
| SD-1.5                | 16.938T  | 859.520M  | 7.032s  | 23.927                    |
| SD-1.5 w/ TGATE       | 9.875T   | 815.557M  | 4.313s  | 20.789                    |
| SD-2.1                | 38.041T  | 865.785M  | 16.121s | 22.609                    |
| SD-2.1 w/ TGATE       | 22.208T  | 815.433 M | 9.878s  | 19.940                    |
| SD-XL                 | 149.438T | 2.570B    | 53.187s | 24.628                    |
| SD-XL w/ TGATE        | 84.438T  | 2.024B    | 27.932s | 22.738                    |
| Pixart-Alpha          | 107.031T | 611.350M  | 61.502s | 38.669                    |
| Pixart-Alpha w/ TGATE | 65.318T  | 462.585M  | 37.867s | 35.825                    |
| DeepCache (SD-XL)     | 57.888T  | -         | 19.931s | 23.755                    |
| DeepCache w/ TGATE    | 43.868T  | -         | 14.666s | 23.999                    |
| LCM (SD-XL)           | 11.955T  | 2.570B    | 3.805s  | 25.044                    |
| LCM w/ TGATE          | 11.171T  | 2.024B    | 3.533s  | 25.028                    |
| LCM (Pixart-Alpha)    | 8.563T   | 611.350M  | 4.733s  | 36.086                    |
| LCM w/ TGATE          | 7.623T   | 462.585M  | 4.543s  | 37.048                    |

The latency is tested on a 1080ti commercial card. 

The MACs and Params are calculated by [calflops](https://github.com/MrYxJ/calculate-flops.pytorch). 

The FID is calculated by [PytorchFID](https://github.com/mseitzer/pytorch-fid).
