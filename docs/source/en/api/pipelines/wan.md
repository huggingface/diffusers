<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Wan

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

<!-- TODO(aryan): update abstract once paper is out -->

## Generating Videos with Wan 2.1

We will first need to install some addtional dependencies.

```shell
pip install -u ftfy imageio-ffmpeg imageio
```

### Text to Video Generation

The following example requires 11GB VRAM to run and uses the smaller `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` model. You can switch it out
for the larger `Wan2.1-I2V-14B-720P-Diffusers` or `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` if you have at least 35GB VRAM available.

```python
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v.mp4", fps=16)
```

<Tip>
You can improve the quality of the generated video by running the decoding step in full precision.
</Tip>

```python
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

# replace this with pipe.to("cuda") if you have sufficient VRAM
pipe.enable_model_cpu_offload()

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

frames = pipe(prompt=prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v.mp4", fps=16)
```

### Image to Video Generation

The Image to Video pipeline requires loading the `AutoencoderKLWan` and the `CLIPVisionModel` components in full precision. The following example will need at least
35GB of VRAM to run.

```python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)

# replace this with pipe.to("cuda") if you have sufficient VRAM
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "wan-i2v.mp4", fps=16)
```

### Video to Video Generation

```python
import torch
from diffusers.utils import load_video, export_to_video
from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline, UniPCMultistepScheduler

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=flow_shift
)
# change to pipe.to("cuda") if you have sufficient VRAM
pipe.enable_model_cpu_offload()

prompt = "A robot standing on a mountain top. The sun is setting in the background"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
)
output = pipe(
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=512,
    guidance_scale=7.0,
    strength=0.7,
).frames[0]

export_to_video(output, "wan-v2v.mp4", fps=16)
```

## Memory Optimizations for Wan 2.1

Base inference with the large 14B Wan 2.1 models can take up to 35GB of VRAM when generating videos at 720p resolution. We'll outline a few memory optimizations we can apply to reduce the VRAM required to run the model.

We'll use `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` model in these examples to demonstrate the memory savings, but the techniques are applicable to all model checkpoints.

### Group Offloading the Transformer and UMT5 Text Encoder

Find more information about group offloading [here](../optimization/memory.md)

#### Block Level Group Offloading

We can reduce our VRAM requirements by applying group offloading to the larger model components of the pipeline; the `WanTransformer3DModel` and `UMT5EncoderModel`. Group offloading will break up the individual modules of a model and offload/onload them onto your GPU as needed during inference. In this example, we'll apply `block_level` offloading, which will group the modules in a model into blocks of size `num_blocks_per_group` and offload/onload them to GPU. Moving to between CPU and GPU does add latency to the inference process. You can trade off between latency and memory savings by increasing or decreasing the `num_blocks_per_group`.

The following example will now only require 14GB of VRAM to run, but will take approximately 30 minutes to generate a video.

```python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanImageToVideoPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, CLIPVisionModel

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)

text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)

transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4,
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)
# Since we've offloaded the larger models alrady, we can move the rest of the model components to GPU
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

max_area = 720 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "wan-i2v.mp4", fps=16)
```

#### Block Level Group Offloading with CUDA Streams

We can speed up group offloading inference, by enabling the use of [CUDA streams](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html). However, using CUDA streams requires moving the model parameters into pinned memory. This allocation is handled by Pytorch under the hood, and can result in a significant spike in CPU RAM usage. Please consider this option if your CPU RAM is atleast 2X the size of the model you are group offloading.

In the following example we will use CUDA streams when group offloading the `WanTransformer3DModel`. When testing on an A100, this example will require 14GB of VRAM, 52GB of CPU RAM, but will generate a video in approximately 9 minutes.

```python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanImageToVideoPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, CLIPVisionModel

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)

text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)

transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)
# Since we've offloaded the larger models alrady, we can move the rest of the model components to GPU
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

max_area = 720 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "wan-i2v.mp4", fps=16)
```

### Applying Layerwise Casting to the Transformer

Find more information about layerwise casting [here](../optimization/memory.md)

In this example, we will model offloading with layerwise casting. Layerwise casting will downcast each layer's weights to `torch.float8_e4m3fn`, temporarily upcast to `torch.bfloat16` during the forward pass of the layer, then revert to `torch.float8_e4m3fn` afterward. This approach reduces memory requirements by approximately 50% while introducing a minor quality reduction in the generated video due to the precision trade-off.

This example will require 20GB of VRAM.

```python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanImageToVideoPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, CLIPVisionModel

model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)
text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")

max_area = 720 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "wan-i2v.mp4", fps=16)
```

## Using a Custom Scheduler

Wan can be used with many different schedulers, each with their own benefits regarding speed and generation quality. By default, Wan uses the `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)` scheduler. You can use a different scheduler as follows:

```python
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline

scheduler_a = FlowMatchEulerDiscreteScheduler(shift=5.0)
scheduler_b = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", scheduler=<CUSTOM_SCHEDULER_HERE>)

# or,
pipe.scheduler = <CUSTOM_SCHEDULER_HERE>
```

## Using Single File Loading with Wan 2.1

The `WanTransformer3DModel` and `AutoencoderKLWan` models support loading checkpoints in their original format via the `from_single_file` loading
method.

```python
import torch
from diffusers import WanPipeline, WanTransformer3DModel

ckpt_path = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
transformer = WanTransformer3DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", transformer=transformer)
```

## Recommendations for Inference
- Keep `AutencoderKLWan` in `torch.float32` for better decoding quality.
- `num_frames` should satisfy the following constraint: `(num_frames - 1) % 4 == 0`
- For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution videos, try higher values (between `7.0` and `12.0`). The default value is `3.0` for Wan.

## WanPipeline

[[autodoc]] WanPipeline
  - all
  - __call__

## WanImageToVideoPipeline

[[autodoc]] WanImageToVideoPipeline
  - all
  - __call__

## WanPipelineOutput

[[autodoc]] pipelines.wan.pipeline_output.WanPipelineOutput
