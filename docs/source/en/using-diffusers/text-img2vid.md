<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Video generation

Video generation models extend image generation (can be considered a 1-frame video) to also process data related to space and time. Making sure all this data - text, space, time - remain consistent and aligned from frame-to-frame is a big challenge in generating long and high-resolution videos.

Modern video models tackle this challenge with the diffusion transformer (DiT) architecture. This reduces computational costs and allows more efficient scaling to larger and higher-quality image and video data.

Check out what some of these video models are capable of below.

<hfoptions id="popular models">
<hfoption id="Wan2.1">

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
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

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
<hfoption id="HunyuanVideo">

```py
import torch
from diffusers importAutoModel, HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

# quantize weights to int4 with bitsandbytes
pipeline_quant_config = PipelineQuantizationConfig(
  quant_backend="bitsandbytes_4bit",
  quant_kwargs={
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
    },
  components_to_quantize="transformer"
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

</hfoption>
<hfoption id="LTX-Video">

```py
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

</hfoption>
</hfoptions>

This guide will cover video generation basics such as which parameters to configure and how to reduce their memory usage.

> [!TIP]
> If you're interested in learning more about how to use a specific model, please refer to their pipeline API model card.

## Pipeline parameters

There are several parameters to configure in the pipeline that'll affect video generation quality or speed. Experimenting with different parameter values is important for discovering the appropriate quality and speed tradeoff.

### num_frames

A frame is a still image that is played in a sequence of other frames to create motion or a video. Control the number of frames generated per second with `num_frames`. Increasing `num_frames` increases perceived motion smoothness and visual coherence, making it especially important for videos with dynamic content. A higher `num_frames` value also increases video duration.

Some video models require more specific `num_frames` values for inference. For example, [`HunyuanVideoPipeline`] recommends calculating the `num_frames` with `(4 * num_frames) +1`. Always check a pipelines API model card to see if there is a recommended value.

```py
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
).to("cuda")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman 
with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The 
camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be 
real-life footage
"""

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

### guidance_scale

Guidance scale or "cfg" controls how closely the generated frames adhere to the input conditioning (text, image or both). Increasing `guidance_scale` generates frames that resemble the input conditions more closely and includes finer details, but risk introducing artifacts and reducing output diversity. Lower `guidance_scale` values encourages looser prompt adherence and increased output variety, but details may not be as great. If it's too low, it may ignore your prompt entirely and generate random noise.

```py
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

pipeline = CogVideoXPipeline.from_pretrained(
  "THUDM/CogVideoX-2b",
  torch_dtype=torch.float16
).to("cuda")

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over
a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, 
with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an 
oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at 
a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
  prompt=prompt,
  guidance_scale=6,
  num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

### negative_prompt

A negative prompt is useful for excluding things you don't want to see in the generated video. It is commonly used to refine the quality and alignment of the generated video by pushing the model away from undesirable elements like "blurry, distorted, ugly". This can create cleaner and more focused videos.

```py
# pip install ftfy
import torch
from diffusers import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

vae = AutoencoderKLWan.from_pretrained(
  "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
pipeline = WanPipeline.from_pretrained(
  "Wan-AI/Wan2.1-T2V-14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(
  pipeline.scheduler.config, flow_shift=5.0
)
pipeline.to("cuda")

pipeline.load_lora_weights("benjamin-paine/steamboat-willie-14b", adapter_name="steamboat-willie")
pipeline.set_adapters("steamboat-willie")

pipeline.enable_model_cpu_offload()

# use "steamboat willie style" to trigger the LoRA
prompt = """
steamboat willie style, golden era animation, The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts 
dynamic shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""

output = pipeline(
  prompt=prompt,
  num_frames=81,
  guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

## Reduce memory usage

Recent video models like [`HunyuanVideoPipeline`] and [`WanPipeline`], which have 10B+ parameters, require a lot of memory and it often exceeds the memory available on consumer hardware. Diffusers offers several techniques for reducing the memory requirements of these large models.

> [!TIP]
> Refer to the [Reduce memory usage](../optimization/memory) guide for more details about other memory saving techniques.

One of these techniques is [group-offloading](../optimization/memory#group-offloading), which offloads groups of internal model layers (such as `torch.nn.Sequential`) to the CPU when it isn't being used. These layers are only loaded when they're needed for computation to avoid storing **all** the model components on the GPU. For a 14B parameter model like [`WanPipeline`], group-offloading can lower the required memory to ~13GB of VRAM.

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
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

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

Another option for reducing memory is to consider quantizing a model, which stores the model weights in a lower precision data type. However, quantization may impact video quality depending on the specific video model. Refer to the quantization [Overivew](../quantization/overview) to learn more about the different supported quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize a model.

```py
# pip install ftfy

import torch
from diffusers import WanPipeline
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import UMT5EncoderModel
from diffusers.utils import export_to_video

# quantize transformer and text encoder weights with bitsandbytes
pipeline_quant_config = PipelineQuantizationConfig(
  quant_backend="bitsandbytes_4bit",
  quant_kwargs={"load_in_4bit": True},
  components_to_quantize=["transformer", "text_encoder"]
)

vae = AutoModel.from_pretrained(
  "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
pipeline = WanPipeline.from_pretrained(
  "Wan-AI/Wan2.1-T2V-14B-Diffusers", vae=vae, quantization_config=pipeline_quant_config, torch_dtype=torch.bfloat16
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(
  pipeline.scheduler.config, flow_shift=5.0
)
pipeline.to("cuda")

pipeline.load_lora_weights("benjamin-paine/steamboat-willie-14b", adapter_name="steamboat-willie")
pipeline.set_adapters("steamboat-willie")

pipeline.enable_model_cpu_offload()

# use "steamboat willie style" to trigger the LoRA
prompt = """
steamboat willie style, golden era animation, The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts 
dynamic shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""

output = pipeline(
  prompt=prompt,
  num_frames=81,
  guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

## Inference speed

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial_.html) can speedup inference by using optimized kernels. Compilation takes longer the first time, but once compiled, it is much faster. It is best to compile the pipeline once, and then use the pipeline multiple times without changing anything. A change, such as in the image size, triggers recompilation.

The example below compiles the transformer in the pipeline and uses the `"max-autotune"` mode to maximize performance.

```py
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

pipeline = CogVideoXPipeline.from_pretrained(
  "THUDM/CogVideoX-2b",
  torch_dtype=torch.float16
).to("cuda")

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. 
The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. 
Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
  prompt=prompt,
  guidance_scale=6,
  num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```