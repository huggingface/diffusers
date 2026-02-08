<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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

# LTX-2

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

LTX-2 is a DiT-based audio-video foundation model designed to generate synchronized video and audio within a single model. It brings together the core building blocks of modern video generation, with open weights and a focus on practical, local execution.

You can find all the original LTX-Video checkpoints under the [Lightricks](https://huggingface.co/Lightricks) organization.

The original codebase for LTX-2 can be found [here](https://github.com/Lightricks/LTX-2).

## Two-stages Generation
Recommended pipeline to achieve production quality generation, this pipeline is composed of two stages:

- Stage 1: Generate a video at the target resolution using diffusion sampling with classifier-free guidance (CFG). This stage produces a coherent low-noise video sequence that respects the text/image conditioning.
- Stage 2: Upsample the Stage 1 output by 2 and refine details using a distilled LoRA model to improve fidelity and visual quality. Stage 2 may apply lighter CFG to preserve the structure from Stage 1 while enhancing texture and sharpness.

Sample usage of text-to-video two stages pipeline

```py
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

device = "cuda:0"
width = 768
height = 512

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Stage 1 default (non-distilled) inference
frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    sigmas=None,
    guidance_scale=4.0,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

# Load Stage 2 distilled LoRA
pipe.load_lora_weights(
    "Lightricks/LTX-2", adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
)
pipe.set_adapters("stage_2_distilled", 1.0)
# VAE tiling is usually necessary to avoid OOM error when VAE decoding
pipe.vae.enable_tiling()
# Change scheduler to use Stage 2 distilled sigmas as is
new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
pipe.scheduler = new_scheduler
# Stage 2 inference with distilled LoRA and sigmas
video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py#L218
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_lora_distilled_sample.mp4",
)
```

## Distilled checkpoint generation
Fastest two-stages generation pipeline using a distilled checkpoint.

```py
import torch
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2Pipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L178
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_distilled_sample.mp4",
)
```

## LTX2Pipeline

[[autodoc]] LTX2Pipeline
  - all
  - __call__

## LTX2ImageToVideoPipeline

[[autodoc]] LTX2ImageToVideoPipeline
  - all
  - __call__

## LTX2LatentUpsamplePipeline

[[autodoc]] LTX2LatentUpsamplePipeline
  - all
  - __call__

## LTX2PipelineOutput

[[autodoc]] pipelines.ltx2.pipeline_output.LTX2PipelineOutput
