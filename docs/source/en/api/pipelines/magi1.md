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

The original repo: https://github.com/SandAI-org/MAGI-1

This model was contributed by [M. Tolga Cangöz](https://github.com/tolgacangoz).

You can find the MAGI-1 checkpoints under the [sand-ai](https://huggingface.co/sand-ai) organization.

The following MAGI-1 models are supported in Diffusers:

**Base Models:**
- [MAGI-1 24B](https://huggingface.co/sand-ai/MAGI-1)
- [MAGI-1 4.5B](https://huggingface.co/sand-ai/MAGI-1-4.5B)

**Distilled Models (faster inference):**
- [MAGI-1 24B Distill](https://huggingface.co/sand-ai/MAGI-1/tree/main/ckpt/magi/24B_distill)
- [MAGI-1 24B Distill+Quant (FP8)](https://huggingface.co/sand-ai/MAGI-1/tree/main/ckpt/magi/24B_distill_quant)
- [MAGI-1 4.5B Distill](https://huggingface.co/sand-ai/MAGI-1/tree/main/ckpt/magi/4.5B_distill)
- [MAGI-1 4.5B Distill+Quant (FP8)](https://huggingface.co/sand-ai/MAGI-1/tree/main/ckpt/magi/4.5B_distill_quant)

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
from diffusers import AutoModel, Magi1Pipeline
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

pipeline = Magi1Pipeline.from_pretrained(
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
from diffusers import AutoModel, Magi1Pipeline
from diffusers.utils import export_to_video
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained("sand-ai/MAGI-1", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("sand-ai/MAGI-1", subfolder="transformer", torch_dtype=torch.bfloat16)

pipeline = Magi1Pipeline.from_pretrained(
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

The example below demonstrates how to use the image-to-video pipeline to generate a video animation from a single image using text prompts for guidance.

<hfoptions id="I2V usage">
<hfoption id="usage">

```python
import torch
from diffusers import Magi1ImageToVideoPipeline, AutoencoderKLMagi1
from diffusers.utils import export_to_video, load_image

model_id = "sand-ai/MAGI-1-I2V"
vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = Magi1ImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load input image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")

prompt = (
    "An astronaut walking on the moon's surface, with the Earth visible in the background. "
    "The astronaut moves slowly in a low-gravity environment, kicking up lunar dust with each step."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, worst quality, low quality"

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,  # Generate 81 frames (~5 seconds at 16fps)
    guidance_scale=5.0,
    num_inference_steps=50,
).frames[0]
export_to_video(output, "astronaut_animation.mp4", fps=16)
```

</hfoption>
</hfoptions>

### Video-to-Video Generation

The example below demonstrates how to use the video-to-video pipeline to extend or continue an existing video using text prompts.

<hfoptions id="V2V usage">
<hfoption id="usage">

```python
import torch
from diffusers import Magi1VideoToVideoPipeline, AutoencoderKLMagi1
from diffusers.utils import export_to_video, load_video

model_id = "sand-ai/MAGI-1-V2V"
vae = AutoencoderKLMagi1.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = Magi1VideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load prefix video (e.g., first 24 frames of a video)
video = load_video("path/to/input_video.mp4", num_frames=24)

prompt = (
    "Continue this video with smooth camera motion and consistent style. "
    "The scene evolves naturally with coherent motion and lighting."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, worst quality, low quality, jumpy motion"

output = pipe(
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,  # Total frames including prefix (24 prefix + 57 generated)
    guidance_scale=5.0,
    num_inference_steps=50,
).frames[0]
export_to_video(output, "video_continuation.mp4", fps=16)
```

</hfoption>
</hfoptions>

## Notes

- MAGI-1 uses autoregressive chunked generation with `chunk_width=6` and `window_size=4`, enabling efficient long video generation.
- The model supports special tokens for quality control (HQ_TOKEN), style (THREE_D_MODEL_TOKEN, TWO_D_ANIME_TOKEN), and motion guidance (STATIC_FIRST_FRAMES_TOKEN, DYNAMIC_FIRST_FRAMES_TOKEN).
- For I2V, the input image is encoded as a clean prefix chunk to condition the video generation.
- For V2V, input video frames (typically 24 frames or ~1.5 seconds) are encoded as clean prefix chunks, and the model generates a continuation.
- MAGI-1 supports LoRAs with [`~loaders.Magi1LoraLoaderMixin.load_lora_weights`].
- Distillation mode can be enabled for faster inference with `enable_distillation=True` (requires distilled model checkpoint).