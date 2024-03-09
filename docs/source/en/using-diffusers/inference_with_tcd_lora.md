<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Performing inference with TCD-LoRA

Trajecotroy Consistency Distillation (TCD) enables the model to generate higher quality, more detailed images with fewer steps. Additionally, TCD demonstrates superior performance even under conditions of high NFEs.

From the [Official Project Page](https://mhh0318.github.io/tcd/), the major merit of TCD can be outlined as follows:

- ***Better than Teacher:*** TCD maintains superior generative quality at both small and large inference steps, even exceeding the performance of [DPM-Solver++(2S)](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver) with Stable Diffusion XL (SDXL). It is worth noting that there is no additional discriminator or LPIPS supervision is included during training.

- ***Flexible NFEs:*** The NFEs for TCD sampling can be varied at will without adversely affecting the quality of the results.

- ***Freely Change the Detailing:*** During inference, the level of detail in the image can be simply modified by adjusing the hyper-parameter gamma. This option does not require any additional parameters.

For more technical details of TCD, please refer to [the paper](https://arxiv.org/abs/2402.19159).

Trajectory consistency distillation can be directly placed on top of a pre-trained diffusion model as a [LoRA](https://huggingface.co/docs/diffusers/main/en/training/lora) module. Such a LoRA can be identified as a versatile acceleration module applicable to different fine-tuned models or LoRAs sharing the same base model without the need for additional training.

TCD-LoRAs are available for [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). 

The corresponding checkpoints can be found at [TCD-SD15](https://huggingface.co/h1t/TCD-SD15-LoRA), [TCD-SD21-base](https://huggingface.co/h1t/TCD-SD21-base-LoRA), and [TCD-SDXL](https://huggingface.co/h1t/TCD-SDXL-LoRA), respectively.


This guide shows how to perform inference with TCD-LoRAs for 
- text-to-image
- inpainting
- community models
- style LoRA
- ControlNet
- IP-Adapter
- AnimateDiff

TCD-LoRA can be considered an advanced method compared with [LCM-LoRA](https://latent-consistency-models.github.io/). The main parts of the TCD-LoRA workflow are as follows::
- Load the task specific pipeline and model.
- Set the scheduler to [`TCDScheduler`].
- Load the TCD-LoRA weights for the model.
- Set the `num_inference_steps` between [4, 50].
- Set `eta` from [0, 1]. Larger `eta` in [`TCDScheduler`] will lead to blurrier images.
- Perform inference with the pipeline with the usual parameters.

Let's look at how we can perform inference with TCD-LoRAs for different tasks.

First, make sure you have [peft](https://github.com/huggingface/peft) installed, for better LoRA support.

```bash
pip install -U peft
```

## Text-to-image

You can use the [`StableDiffusionXLPipeline`] with the scheduler: [`TCDScheduler`] and then load the TCD-LoRA. Together with the TCD-LoRA and the TCDScheduler, the pipeline enables a fast inference workflow with high quality outputs.

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


<Tip>

Eta (referred to as `gamma` in the paper) is used to control the stochasticity in every step.
A value of 0.3 often yields good results, where eta = 0 means determinstic and eta = 1 is identity to Multi-step Consistency Sampler (as well as LCMScheduler).
We recommend using a higher eta when increasing the number of inference steps.

</Tip>

## TCD-LoRA is Versatile for Community Models

As mentioned above, the TCD-LoRA is versatile for community models and plugins. To test-drive this, load a community fine-tuned base model [animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0).

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

Furthermore, TCD-LoRA also supports LoRAs corresponding to other styles. Below is an example with [Papercut](https://huggingface.co/TheLastBen/Papercut_SDXL). To learn more about how to combine LoRAs, refer to [this guide](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#combine-multiple-adapters).

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


## Inpainting with TCD 


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
  eta=0.3, # Eta (referred to as `gamma` in the paper) is used to control the stochasticity in every step. A value of 0.3 often yields good results.
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/inpainting_tcd.png)


## Compatibility with ControlNet

For this example, you'll keep using the SDXL model and the TCD-LoRA for SDXL with depth and canny ControlNets.

### Depth ControlNet
```python
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid
from scheduling_tcd import TCDScheduler 

device = "cuda"
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

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
).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)
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
).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)
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

## IP-Adapter

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



## AnimateDiff

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