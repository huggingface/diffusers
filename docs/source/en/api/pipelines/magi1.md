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

# MAGI-1

[MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/abs/2505.13211) by Sand.ai.

*MAGI-1 is an autoregressive video generation model that generates videos chunk-by-chunk instead of as a whole. Each chunk (24 frames) is denoised holistically, and the generation of the next chunk begins as soon as the current one reaches a certain level of denoising. This pipeline design enables concurrent processing of up to four chunks for efficient video generation. The model leverages a specialized architecture with a transformer-based VAE with 8x spatial and 4x temporal compression, and a diffusion transformer with several key innovations including Block-Causal Attention, Parallel Attention Block, QK-Norm and GQA, Sandwich Normalization in FFN, SwiGLU, and Softcap Modulation.*

You can find the MAGI-1 checkpoints under the [sand-ai](https://huggingface.co/sand-ai) organization.

The following MAGI models are supported in Diffusers:
- [MAGI-1 24B](https://huggingface.co/sand-ai/MAGI-1)
- [MAGI-1 4.5B](https://huggingface.co/sand-ai/MAGI-1-4.5B)

> [!TIP]
> Click on the MAGI-1 models in the right sidebar for more examples of video generation.

### Text-to-Video Generation

The example below demonstrates how to generate a video from text optimized for memory or inference speed.

<hfoptions id="T2V usage">
<hfoption id="T2V memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The MAGI-1 text-to-video model below requires ~13GB of VRAM.

```py
import torch
import numpy as np
from diffusers import AutoModel, MagiPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained("sand-ai/MAGI-1", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="transformer", torch_dtype=torch.bfloat16)

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

pipeline = MagiPipeline.from_pretrained(
    "sand-ai/MAGI-1",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = """
A majestic eagle soaring over a mountain landscape. The eagle's wings are spread wide,
catching the golden sunlight as it glides through the clear blue sky. Below, snow-capped
mountains stretch to the horizon, with pine forests and a winding river visible in the valley.
"""
negative_prompt = """
Poor quality, blurry, pixelated, low resolution, distorted proportions, unnatural colors,
watermark, text overlay, incomplete rendering, glitches, artifacts, unrealistic lighting
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=24,
    guidance_scale=7.0,
).frames[0]
export_to_video(output, "output.mp4", fps=8)
```

</hfoption>
<hfoption id="T2V inference speed">

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster.

```py
import torch
import numpy as np
from diffusers import AutoModel, MagiPipeline
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained("sand-ai/MAGI-1", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="transformer", torch_dtype=torch.bfloat16)

pipeline = MagiPipeline.from_pretrained(
    "sand-ai/MAGI-1",
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
A majestic eagle soaring over a mountain landscape. The eagle's wings are spread wide,
catching the golden sunlight as it glides through the clear blue sky. Below, snow-capped
mountains stretch to the horizon, with pine forests and a winding river visible in the valley.
"""
negative_prompt = """
Poor quality, blurry, pixelated, low resolution, distorted proportions, unnatural colors,
watermark, text overlay, incomplete rendering, glitches, artifacts, unrealistic lighting
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=24,
    guidance_scale=7.0,
).frames[0]
export_to_video(output, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

### Image-to-Video Generation

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text description and a starting frame.

<hfoptions id="I2V usage">
<hfoption id="usage">

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLMagi1, MagiImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

model_id = "sand-ai/MAGI-1"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = MagiImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image.png")

def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

image, height, width = aspect_ratio_resize(image, pipe)

prompt = "A beautiful landscape with mountains and a lake. The camera slowly pans from left to right, revealing more of the landscape."

output = pipe(
    image=image, prompt=prompt, height=height, width=width, guidance_scale=7.5, num_frames=24
).frames[0]
export_to_video(output, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

### First-Last-Frame-to-Video Generation

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text description, a starting frame, and an ending frame.

<hfoptions id="FLF2V usage">
<hfoption id="usage">

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLMagi1, MagiImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

model_id = "sand-ai/MAGI-1"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = MagiImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

first_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/first_frame.png")
last_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/last_frame.png")

def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

def center_crop_resize(image, height, width):
    # Calculate resize ratio to match first frame dimensions
    resize_ratio = max(width / image.width, height / image.height)

    # Resize the image
    width = round(image.width * resize_ratio)
    height = round(image.height * resize_ratio)
    size = [width, height]
    image = TF.center_crop(image, size)

    return image, height, width

first_frame, height, width = aspect_ratio_resize(first_frame, pipe)
if last_frame.size != first_frame.size:
    last_frame, _, _ = center_crop_resize(last_frame, height, width)

prompt = "A car driving down a winding mountain road. The camera follows the car as it navigates the curves, revealing beautiful mountain scenery in the background."

output = pipe(
    image=first_frame, last_image=last_frame, prompt=prompt, height=height, width=width, guidance_scale=7.5, num_frames=24
).frames[0]
export_to_video(output, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

### Video-to-Video Generation

The example below demonstrates how to use the video-to-video pipeline to generate a video based on an existing video and text prompt.

<hfoptions id="V2V usage">
<hfoption id="usage">

```python
import torch
import numpy as np
from diffusers import AutoencoderKLMagi1, MagiVideoToVideoPipeline
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel

model_id = "sand-ai/MAGI-1"
text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = MagiVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, text_encoder=text_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load input video
video_path = "input_video.mp4"
video = load_video(video_path)

prompt = "Convert this video to an anime style with vibrant colors and exaggerated features"
negative_prompt = "Poor quality, blurry, distorted, unrealistic lighting, bad composition"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    video=video,
    strength=0.7,  # Controls how much to preserve from original video
    guidance_scale=7.5,
).frames[0]
export_to_video(output, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

## Notes

- MAGI-1 supports LoRAs with [`~loaders.MagiLoraLoaderMixin.load_lora_weights`].