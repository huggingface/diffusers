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

[LTX-2](https://hf.co/papers/2601.03233) is a DiT-based foundation model designed to generate synchronized video and audio within a single model. It brings together the core building blocks of modern video generation, with open weights and a focus on practical, local execution.

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

## Condition Pipeline Generation

You can use `LTX2ConditionPipeline` to specify image and/or video conditions at arbitrary latent indices. For example, we can specify both a first-frame and last-frame condition to perform first-last-frame-to-video (FLF2V) generation:

```py
import torch
from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
    "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
    "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
    "low-angle perspective."
)

first_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png",
)
last_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png",
)
first_cond = LTX2VideoCondition(frames=first_image, index=0, strength=1.0)
last_cond = LTX2VideoCondition(frames=last_image, index=-1, strength=1.0)
conditions = [first_cond, last_cond]

frame_rate = 24.0
video_latent, audio_latent = pipe(
    conditions=conditions,
    prompt=prompt,
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
    width=width * 2,
    height=height * 2,
    num_inference_steps=3,
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
    output_path="ltx2_distilled_flf2v.mp4",
)
```

You can use both image and video conditions:

```py
import torch
from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image, load_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is "
    "divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features "
    "dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered "
    "clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, "
    "with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The "
    "landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the "
    "solitude and beauty of a winter drive through a mountainous region."
)

cond_video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)
cond_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
)
video_cond = LTX2VideoCondition(frames=cond_video, index=0, strength=1.0)
image_cond = LTX2VideoCondition(frames=cond_image, index=8, strength=1.0)
conditions = [video_cond, image_cond]

frame_rate = 24.0
video, audio = pipe(
    conditions=conditions,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_cond_video.mp4",
)
```

Because the conditioning is done via latent frames, the 8 data space frames corresponding to the specified latent frame for an image condition will tend to be static.

## Multimodal Guidance

LTX-2.X pipelines support multimodal guidance. It is composed of three terms, all using a CFG-style update rule:

1. Classifier-Free Guidance (CFG): standard [CFG](https://huggingface.co/papers/2207.12598) where the perturbed ("weaker") output is generated using the negative prompt.
2. Spatio-Temporal Guidance (STG): [STG](https://huggingface.co/papers/2411.18664) moves away from a perturbed output created from short-cutting self-attention operations and substitutes in the attention values instead. The idea is that this creates sharper videos and better spatiotemporal consistency.
3. Modality Isolation Guidance: moves away from a perturbed output created from disabling cross-modality (audio-to-video and video-to-audio) cross attention. This guidance is more specific to [LTX-2.X](https://huggingface.co/papers/2601.03233) models, with the idea that this produces better consistency between the generated audio and video.

These are controlled by the `guidance_scale`, `stg_scale`, and `modality_scale` arguments and can be set separately for video and audio. Additionally, for STG the transformer block indices where self-attention is skipped needs to be specified via the `spatio_temporal_guidance_blocks` argument. The LTX-2.X pipelines also support [guidance rescaling](https://huggingface.co/papers/2305.08891) to help reduce over-exposure, which can be a problem when the guidance scales are set to high values.

```py
import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "dg845/LTX-2.3-Diffusers"

pipe = LTX2ImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)

video, audio = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,  # Recommended LTX-2.3 guidance parameters
    stg_scale=1.0,  # Note that 0.0 (not 1.0) means that STG is disabled (all other guidance is disabled at 1.0)
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,  # Note that a higher CFG guidance scale is recommended for audio
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_i2v_stage_1.mp4",
)
```

## Prompt Enhancement

The LTX-2.X models are sensitive to prompting style. Refer to the [official prompting guide](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) for recommendations on how to write a good prompt. Using prompt enhancement, where the supplied prompts are enhanced using the pipeline's text encoder (by default a [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) model) given a system prompt, can also improve sample quality. The optional `processor` pipeline component needs to be present to use prompt enhancement. Enable prompt enhancement by supplying a `system_prompt` argument:


```py
import torch
from transformers import Gemma3Processor
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, T2V_DEFAULT_SYSTEM_PROMPT

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "dg845/LTX-2.3-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()
if getattr(pipe, "processor", None) is None:
    processor = Gemma3Processor.from_pretrained("google/gemma-3-12b-it-qat-q4_0-unquantized")
    pipe.processor = processor

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    system_prompt=T2V_DEFAULT_SYSTEM_PROMPT,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_t2v_stage_1.mp4",
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

## LTX2ConditionPipeline

[[autodoc]] LTX2ConditionPipeline
  - all
  - __call__

## LTX2LatentUpsamplePipeline

[[autodoc]] LTX2LatentUpsamplePipeline
  - all
  - __call__

## LTX2PipelineOutput

[[autodoc]] pipelines.ltx2.pipeline_output.LTX2PipelineOutput
