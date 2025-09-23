<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TCDScheduler

[Trajectory Consistency Distillation](https://huggingface.co/papers/2402.19159) by Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao and Tat-Jen Cham introduced a Strategic Stochastic Sampling (Algorithm 4) that is capable of generating good samples in a small number of steps. Distinguishing it as an advanced iteration of the multistep scheduler (Algorithm 1) in the [Consistency Models](https://huggingface.co/papers/2303.01469), Strategic Stochastic Sampling specifically tailored for the trajectory consistency function.

The abstract from the paper is:

*Latent Consistency Model (LCM) extends the Consistency Model to the latent space and leverages the guided consistency distillation technique to achieve impressive performance in accelerating text-to-image synthesis. However, we observed that LCM struggles to generate images with both clarity and detailed intricacy. To address this limitation, we initially delve into and elucidate the underlying causes. Our investigation identifies that the primary issue stems from errors in three distinct areas. Consequently, we introduce Trajectory Consistency Distillation (TCD), which encompasses trajectory consistency function and strategic stochastic sampling. The trajectory consistency function diminishes the distillation errors by broadening the scope of the self-consistency boundary condition and endowing the TCD with the ability to accurately trace the entire trajectory of the Probability Flow ODE. Additionally, strategic stochastic sampling is specifically designed to circumvent the accumulated errors inherent in multi-step consistency sampling, which is meticulously tailored to complement the TCD model. Experiments demonstrate that TCD not only significantly enhances image quality at low NFEs but also yields more detailed results compared to the teacher model at high NFEs.*

The original codebase can be found at [jabir-zheng/TCD](https://github.com/jabir-zheng/TCD).

## General tasks

In this guide, let's use the [`StableDiffusionXLPipeline`] and the [`TCDScheduler`]. Use the [`~StableDiffusionPipeline.load_lora_weights`] method to load the SDXL-compatible TCD-LoRA weights.

A few tips to keep in mind for TCD-LoRA inference are to:

- Keep the `num_inference_steps` between 4 and 50
- Set `eta` (used to control stochasticity at each step) between 0 and 1. You should use a higher `eta` when increasing the number of inference steps, but the downside is that a larger `eta` in [`TCDScheduler`] leads to blurrier images. A value of 0.3 is recommended to produce good results.

<hfoptions id="tasks">
<hfoption id="text-to-image">

```python
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/demo_image.png)

</hfoption>

<hfoption id="inpainting">

```python
import torch
from diffusers import AutoPipelineForInpainting, TCDScheduler
from diffusers.utils import load_image, make_image_grid

device = "cuda"
base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = AutoPipelineForInpainting.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a tiger sitting on a park bench"

image = pipe(
  prompt=prompt,
  image=init_image,
  mask_image=mask_image,
  num_inference_steps=8,
  guidance_scale=0,
  eta=0.3,
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/inpainting_tcd.png)


</hfoption>
</hfoptions>

## Community models

TCD-LoRA also works with many community finetuned models and plugins. For example, load the [animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0) checkpoint which is a community finetuned version of SDXL for generating anime images.

```python
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base_model_id = "cagliostrolab/animagine-xl-3.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "A man, clad in a meticulously tailored military uniform, stands with unwavering resolve. The uniform boasts intricate details, and his eyes gleam with determination. Strands of vibrant, windswept hair peek out from beneath the brim of his cap."

image = pipe(
    prompt=prompt,
    num_inference_steps=8,
    guidance_scale=0,
    eta=0.3,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/animagine_xl.png)

TCD-LoRA also supports other LoRAs trained on different styles. For example, let's load the [TheLastBen/Papercut_SDXL](https://huggingface.co/TheLastBen/Papercut_SDXL) LoRA and fuse it with the TCD-LoRA with the [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] method.

> [!TIP]
> Check out the [Merge LoRAs](merge_loras) guide to learn more about efficient merging methods.

```python
import torch
from diffusers import StableDiffusionXLPipeline
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"
styled_lora_id = "TheLastBen/Papercut_SDXL"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id, adapter_name="tcd")
pipe.load_lora_weights(styled_lora_id, adapter_name="style")
pipe.set_adapters(["tcd", "style"], adapter_weights=[1.0, 1.0])

prompt = "papercut of a winter mountain, snow"

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/styled_lora.png)


## Adapters

TCD-LoRA is very versatile, and it can be combined with other adapter types like ControlNets, IP-Adapter, and AnimateDiff.

<hfoptions id="adapters">
<hfoption id="ControlNet">

### Depth ControlNet

```python
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid
from scheduling_tcd import TCDScheduler

device = "cuda"
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "stormtrooper lecture, photorealistic"

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
depth_image = get_depth_map(image)

controlnet_conditioning_scale = 0.5  # recommended for good generalization

image = pipe(
    prompt,
    image=depth_image,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([depth_image, image], rows=1, cols=2)
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_depth_tcd.png)

### Canny ControlNet
```python
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "ultrarealistic shot of a furry blue bird"

canny_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

image = pipe(
    prompt,
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([canny_image, image], rows=1, cols=2)
```
![](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_canny_tcd.png)

<Tip>
The inference parameters in this example might not work for all examples, so we recommend you to try different values for `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale` and `cross_attention_kwargs` parameters and choose the best one.
</Tip>

</hfoption>
<hfoption id="IP-Adapter">

This example shows how to use the TCD-LoRA with the [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/tree/main) and SDXL.

```python
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid

from ip_adapter import IPAdapterXL
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

ref_image = load_image("https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png").resize((512, 512))

prompt = "best quality, high quality, wearing sunglasses"

image = ip_model.generate(
    pil_image=ref_image,
    prompt=prompt,
    scale=0.5,
    num_samples=1,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    seed=0,
)[0]

grid_image = make_image_grid([ref_image, image], rows=1, cols=2)
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/ip_adapter.png)



</hfoption>
<hfoption id="AnimateDiff">

[`AnimateDiff`] allows animating images using Stable Diffusion models. TCD-LoRA can substantially accelerate the process without degrading image quality. The quality of animation with TCD-LoRA and AnimateDiff has a more lucid outcome.

```python
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from scheduling_tcd import TCDScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
).to("cuda")

# set TCDScheduler
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# load TCD LoRA
pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="tcd")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora")

pipe.set_adapters(["tcd", "motion-lora"], adapter_weights=[1.0, 1.2])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual_seed(0)
frames = pipe(
    prompt=prompt,
    num_inference_steps=5,
    guidance_scale=0,
    cross_attention_kwargs={"scale": 1},
    num_frames=24,
    eta=0.3,
    generator=generator
).frames[0]
export_to_gif(frames, "animation.gif")
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/animation_example.gif)

</hfoption>
</hfoptions>

## TCDScheduler
[[autodoc]] TCDScheduler


## TCDSchedulerOutput
[[autodoc]] schedulers.scheduling_tcd.TCDSchedulerOutput

