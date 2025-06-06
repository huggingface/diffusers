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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
      <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
    </a>
  </div>
</div>

# Wan2.1

[Wan2.1](https://files.alicdn.com/tpsservice/5c9de1c74de03972b7aa657e5a54756b.pdf) is a series of large diffusion transformer available in two versions, a high-performance 14B parameter model and a more accessible 1.3B version. Trained on billions of images and videos, it supports tasks like text-to-video (T2V) and image-to-video (I2V) while enabling features such as camera control and stylistic diversity. The Wan-VAE features better image data compression and a feature cache mechanism that encodes and decodes a video in chunks. To maintain continuity, features from previous chunks are cached and reused for processing subsequent chunks. This improves inference efficiency by reducing memory usage. Wan2.1 also uses a multilingual text encoder and the diffusion transformer models space and time relationships and text conditions with each time step to capture more complex video dynamics.

You can find all the original Wan2.1 checkpoints under the [Wan-AI](https://huggingface.co/Wan-AI) organization.

> [!TIP]
> Click on the Wan2.1 models in the right sidebar for more examples of video generation.

The example below demonstrates how to generate a video from text optimized for memory or inference speed.

<hfoptions id="usage">
<hfoption id="memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The Wan2.1 text-to-video model below requires ~13GB of VRAM.

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
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
<hfoption id="inference speed">

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster.

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

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

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
</hfoptions>

## Notes

- Wan2.1 supports LoRAs with [`~loaders.WanLoraLoaderMixin.load_lora_weights`].

  <details>
  <summary>Show example code</summary>

  ```py
  # pip install ftfy
  import torch
  from diffusers import AutoModel, WanPipeline
  from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
  from diffusers.utils import export_to_video

  vae = AutoModel.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32
  )
  pipeline = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae=vae, torch_dtype=torch.bfloat16
  )
  pipeline.scheduler = UniPCMultistepScheduler.from_config(
      pipeline.scheduler.config, flow_shift=5.0
  )
  pipeline.to("cuda")

  pipeline.load_lora_weights("benjamin-paine/steamboat-willie-1.3b", adapter_name="steamboat-willie")
  pipeline.set_adapters("steamboat-willie")

  pipeline.enable_model_cpu_offload()

  # use "steamboat willie style" to trigger the LoRA
  prompt = """
  steamboat willie style, golden era animation, The camera rushes from far to near in a low-angle shot, 
  revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
  for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
  Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
  shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
  """

  output = pipeline(
      prompt=prompt,
      num_frames=81,
      guidance_scale=5.0,
  ).frames[0]
  export_to_video(output, "output.mp4", fps=16)
  ```

  </details>

- [`WanTransformer3DModel`] and [`AutoencoderKLWan`] supports loading from single files with [`~loaders.FromSingleFileMixin.from_single_file`].

  <details>
  <summary>Show example code</summary>

  ```py
  # pip install ftfy
  import torch
  from diffusers import WanPipeline, AutoModel

  vae = AutoModel.from_single_file(
      "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors"
  )
  transformer = AutoModel.from_single_file(
      "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
      torch_dtype=torch.bfloat16
  )
  pipeline = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      vae=vae,
      transformer=transformer,
      torch_dtype=torch.bfloat16
  )
  ```

  </details>

- Set the [`AutoencoderKLWan`] dtype to `torch.float32` for better decoding quality.

- The number of frames per second (fps) or `k` should be calculated by `4 * k + 1`.

- Try lower `shift` values (`2.0` to `5.0`) for lower resolution videos and higher `shift` values (`7.0` to `12.0`) for higher resolution images.

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