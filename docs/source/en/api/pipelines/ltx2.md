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

The shared `LTX2Pipeline` / `LTX2ImageToVideoPipeline` `__call__` defaults match the LTX-2.5 reference (`num_inference_steps=30`; `num_frames` is optional when a `duration_head` is present, otherwise it falls back to `121`). The examples below use those defaults for LTX-2.0/2.3 as well.

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
from diffusers.utils import encode_video

device = "cuda:0"
width = 768
height = 512

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", dtype=torch.bfloat16
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
    num_inference_steps=30,
    sigmas=None,
    guidance_scale=3.0,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
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
from diffusers.utils import encode_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2Pipeline.from_pretrained(
    model_path, dtype=torch.bfloat16
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
    dtype=torch.bfloat16,
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
from diffusers.utils import encode_video
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
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
    dtype=torch.bfloat16,
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
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image, load_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
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
    num_inference_steps=30,
    guidance_scale=3.0,
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
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2ImageToVideoPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
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

The LTX-2.X models are sensitive to prompting style. Refer to the [official prompting guide](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) for recommendations on how to write a good prompt. Using prompt enhancement, where the supplied prompts are enhanced using the pipeline's text encoder (by default a [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) model) given a system prompt, can also improve sample quality. The optional `processor` pipeline component needs to be present to use prompt enhancement. Enable it with `enable_prompt_enhancement=True` and a `system_prompt` (opt-in, matching the Lightricks reference pipelines):


```py
import torch
from transformers import Gemma3Processor
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, T2V_DEFAULT_SYSTEM_PROMPT

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
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
    enable_prompt_enhancement=True,
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

## LTX-2.5

LTX-2.5 reuses the same `LTX2Pipeline`/`LTX2VideoTransformer3DModel`/`AutoencoderKLLTX2Video`/etc. classes as LTX-2.3 — there is no separate pipeline class for it. The user-visible difference is the text encoder: LTX-2.5 is paired with a Gemma 4 (`gemma4_unified`) checkpoint instead of Gemma 3. This is loaded automatically when you call `from_pretrained` on a converted LTX-2.5 checkpoint (via the `transformers` `Auto*` classes), so no extra setup is needed at inference time — just point `from_pretrained` at an LTX-2.5 repo instead of an LTX-2.3 one.

[`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) ships both transformers: the **distilled** DiT in `transformer/`, which is what `model_index.json` points at, and the full/SFT DiT in `transformer_full/`, which has to be loaded explicitly (see [Full / SFT transformer](#full--sft-transformer)). The repo's `scheduler/` is configured for the distilled checkpoint (`use_dynamic_shifting=False`, `shift_terminal=None`) so that its sigma schedule is used exactly as given. Everything [two-stage generation](#two-stage-generation-for-ltx-25) needs is shipped there too: a `latent_upsampler/` subfolder and the stage 2 distilled LoRA, `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`, at the root of the repo.

Distilled inference is driven by an explicit sigma schedule rather than a step count, and runs unguided (`guidance_scale=1.0`, so `negative_prompt` is unused). Passing `num_inference_steps` instead would hand the model a generic linear schedule and quietly cost quality:

```py
import torch
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "Lightricks/LTX-2.5-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

video, audio = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v.mp4",
)
```

### Two-stage generation for LTX-2.5

LTX-2.5 supports both two-stage variants, and `DISTILLED_SIGMA_VALUES` / `STAGE_2_DISTILLED_SIGMA_VALUES` are its reference schedules:

- **Distilled checkpoint, both stages** — the reference recipe for the default `transformer/`, and the one shown below. No stage 2 LoRA is involved, since the transformer is already distilled; this is [Distilled checkpoint generation](#distilled-checkpoint-generation) with LTX-2.5 weights.
- **Full/SFT stage 1 + distilled LoRA stage 2** — [Two-stages Generation](#two-stages-generation) as described at the top of this page, using `transformer_full/` and the shipped LoRA. See [below](#stage-2-with-the-distilled-lora) for what changes.

Stage 1 runs at half the target resolution, the upsampler doubles it, and stage 2 refines at full resolution — video *and* audio, both reseeded from the stage 1 latents at `noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0]`. Height and width must be divisible by 64, since stage 1 halves each axis and still has to land on the VAE's spatial grid.

```py
import torch
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"
width = 1536
height = 1024
num_frames = 121
frame_rate = 24.0
model_path = "Lightricks/LTX-2.5-Diffusers"

# One generator for the whole call, threaded through both stages, so stage 2 continues the noise
# stream instead of repeating stage 1's draw.
generator = torch.Generator(device).manual_seed(42)

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

# Stage 1: half resolution, 8 distilled sigmas
video_latent, audio_latent = pipe(
    prompt=prompt,
    width=width // 2,
    height=height // 2,
    num_frames=num_frames,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
# `latents_normalized=False`: `output_type="latent"` already applied the latent statistics, and the
# upsampler is trained on denormalized latents. Stage 2 renormalizes them in `prepare_latents`.
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    latents_normalized=False,
    output_type="latent",
    return_dict=False,
)[0]

# Stage 2: full resolution, 3 sigmas, reseeded from stage 1. Pass `num_frames` explicitly here --
# omitting it would run the duration head a second time instead of using the stage 1 length.
pipe.vae.enable_tiling()
video, audio = pipe(
    prompt=prompt,
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],  # renoise with the stage 2 entry sigma
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_two_stages.mp4",
)
```

When the length comes from the [duration head](#automatic-duration-for-ltx-25) rather than an explicit `num_frames`, let stage 1 decide and recover the realized length from its latents (`[B, C, F, H, W]`) before stage 2 runs, instead of predicting a second time:

```py
num_frames = (video_latent.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
```

#### Stage 2 with the distilled LoRA

To run [Two-stages Generation](#two-stages-generation) instead — full/SFT DiT for stage 1, distilled LoRA for stage 2 — build the pipeline as in [Full / SFT transformer](#full--sft-transformer) and generate stage 1 latents with that guidance stack. Two things then differ from LTX-2.0/2.3. The LoRA lives in the diffusers repo itself rather than alongside the original weights, and the scheduler flip goes the other way round: LTX-2.5 ships the *distilled* scheduler config, so stage 1 is what turned dynamic shifting on, and stage 2 turns it back off.

```py
pipe.load_lora_weights(
    "Lightricks/LTX-2.5-Diffusers",
    adapter_name="stage_2_distilled",
    weight_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
)
pipe.set_adapters("stage_2_distilled", 1.0)
pipe.vae.enable_tiling()

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
```

The upsample step and the stage 2 call itself are unchanged from the distilled recipe above: same `sigmas=STAGE_2_DISTILLED_SIGMA_VALUES`, same `noise_scale`, and `guidance_scale=1.0`, since stage 2 is running a distilled model either way.

### Convolutional and diffusion decoding

LTX-2.5 ships two video decoders over the same latent space, so latents are interchangeable between them:

- `vae/` — the convolutional VAE ([`AutoencoderKLLTX2Video`]). It is what the pipelines decode with, so every snippet above already uses it, and it is the only one of the two that tiles (`pipe.vae.enable_tiling()`), which is usually what makes a high resolution fit.
- `diffusion_decoder/` — [`LTX2VideoDiffusionDecoderModel`]. It is a diffusion model in its own right rather than a pipeline component, so it is not passed as a `vae`: run the pipeline with `output_type="latent"` and hand the latents to [`LTX2VideoDiffusionDecodePipeline`].

Encoding always goes through `vae/`, so image and video conditioning are unaffected by the choice.

Two things change when you decode with the diffusion decoder. `output_type="latent"` also skips the vocoder, so the audio comes back as latents and has to be finished by hand, and the NATTEN processor is effectively required at video resolutions:

```py
import torch
from diffusers import LTX2Pipeline, LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(42)
model_path = "Lightricks/LTX-2.5-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

latents, audio_latents = pipe(
    prompt=prompt,
    width=960,
    height=544,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

# `output_type="latent"` skips the vocoder, so finish the audio by hand. These latents are already
# denormalized, which is what `audio_vae.decode` expects.
mel = pipe.audio_vae.decode(audio_latents.to(pipe.audio_vae.dtype), return_dict=False)[0]
audio = pipe.vocoder(mel)

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    model_path, subfolder="diffusion_decoder", dtype=torch.bfloat16
).to(device)
# The decoder runs on the `flex` backend by default, and uncompiled `flex_attention` materializes the
# full score matrix -- tens of GB at video resolutions. NATTEN's kernels are what the original
# implementation uses; they are fetched from the Hub by `kernels` (`pip install kernels`), not from a
# local NATTEN build. Switching the attention *backend* instead raises: only `flex` takes the BlockMask.
decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
# Decode in overlapping tiles so peak memory scales with the tile size rather than the video size.
decoder.enable_tiling()

decode_pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=pipe.scheduler)

# `denormalize=False`: `output_type="latent"` already applied the latent statistics, so applying them
# again would rescale every channel by its std a second time. The decoder draws the noise it denoises,
# so pass a generator to make decoding reproducible.
video = decode_pipe(
    latents, generator=generator, output_type="np", denormalize=False, return_dict=False
)[0]

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_diffusion_decode.mp4",
)
```

To combine this with [two-stage generation](#two-stage-generation-for-ltx-25), ask *stage 2* for `output_type="latent"` and decode that.

`decoder.enable_tiling()` is what keeps a high resolution decode in memory, the same way `pipe.vae.enable_tiling()` does for the convolutional VAE. The memory-dominant part of the decode — the last upsampling stage and the diffusion stage — then runs on overlapping tiles that are blended back together, so peak memory is bounded by the tile size instead of the video size. Tiling only kicks in once the latent exceeds one tile, and the tile and overlap sizes can be tuned via the `tile_sample_min_*` / `tile_sample_stride_*` arguments (defaults match the reference implementation). Since the diffusion stage denoises each tile separately, a tiled decode does not reproduce the untiled result exactly.

On a single card it is also worth moving the pipeline out of the way before decoding (`pipe.to("cpu")` and `torch.cuda.empty_cache()`, after capturing `pipe.scheduler` and the vocoder's `output_sampling_rate`), since the decoder needs its own headroom. See [`LTX2VideoDiffusionDecoderModel`] for the attention backends, the tiling details, and the rest of the decoder's behaviour.

### Full / SFT transformer

`transformer_full/` is not referenced by `model_index.json`, so load it explicitly. It also needs a different scheduler and a real guidance stack: the shipped `scheduler/` is configured for the distilled checkpoint, and the guidance defaults are LTX-2.0-era generics that leave an LTX-2.5 SFT run visibly under-guided without raising anything. The [Multimodal Guidance](#multimodal-guidance) recommendations apply here unchanged, including STG on block `28`:

```py
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

device = "cuda"
model_path = "Lightricks/LTX-2.5-Diffusers"

# Passing `transformer=` keeps `from_pretrained` from fetching the distilled folder as well.
transformer = LTX2VideoTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer_full", dtype=torch.bfloat16
)
pipe = LTX2Pipeline.from_pretrained(model_path, transformer=transformer, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

# Re-enable dynamic shifting and the terminal shift, which the distilled configuration turns off.
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1
)

video, audio = pipe(
    prompt="A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
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
    generator=torch.Generator(device).manual_seed(42),
    output_type="np",
    return_dict=False,
)
```

Drop `sigmas` here — the full DiT takes its schedule from the scheduler.

### Prompt Enhancement for LTX-2.5

**Using prompt enhancement is strongly recommended for LTX-2.5; pass `enable_prompt_enhancement=True` to opt in** (same as the Lightricks reference pipelines). Unlike LTX-2.0/2.3, where the same text encoder checkpoint doubles as the enhancer (see [Prompt Enhancement](#prompt-enhancement) above), LTX-2.5's fine-tuned text encoder was not trained for enhancement. Instead, enhancement uses a separate, off-the-shelf `google/gemma-4-E2B-it` checkpoint. Load it into the pipeline's optional `prompt_enhancer`/`processor` components, then enable enhancement — the pipeline defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and the Gemma 4 recipe (`do_sample=False`, `no_repeat_ngram_size=5`, `max_new_tokens=600`). Pass an explicit `system_prompt=` to override:

```py
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "Lightricks/LTX-2.5-Diffusers"
enhancer_model_id = "google/gemma-4-E2B-it"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()
if getattr(pipe, "prompt_enhancer", None) is None:
    pipe.prompt_enhancer = AutoModelForImageTextToText.from_pretrained(enhancer_model_id)
    pipe.processor = AutoProcessor.from_pretrained(enhancer_model_id)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

video, audio = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    enable_prompt_enhancement=True,
    # No `system_prompt=` needed -- defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` when `prompt_enhancer` is set.
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_enhanced.mp4",
)
```

The same applies to image-to-video with `LTX2ImageToVideoPipeline`: set `pipe.prompt_enhancer`/`pipe.processor` the same way and pass `enable_prompt_enhancement=True` (using `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`, conditioning on both the reference image and the text prompt) — again, no `system_prompt=` needed unless you want to override it.

### Automatic duration for LTX-2.5

LTX-2.5 checkpoints ship a small `duration_head` that predicts how long the described shot should be, from the same text-connector output the transformer is conditioned on. When the loaded pipeline has one, **`num_frames` is auto-predicted by default** — omit it and the model chooses the length:

```py
video, audio = pipe(prompt=prompt, output_type="np", return_dict=False)
```

To set the length yourself, pass `num_frames` explicitly. An integer always wins over the head:

```py
video, audio = pipe(prompt=prompt, num_frames=121, output_type="np", return_dict=False)
```

Pipelines loaded from LTX-2.0 or LTX-2.3 checkpoints have no duration head and keep the previous default of 121 frames, so this changes nothing for them.

Pass `min_seconds` / `max_seconds` to constrain the prediction. The raw prediction is clamped into the range, then converted to frames:

```py
video, audio = pipe(
    prompt=prompt,
    min_seconds=2.0,
    max_seconds=10.0,
    frame_rate=frame_rate,
    output_type="np",
    return_dict=False,
)
```

Predicted frame counts are snapped to the VAE's causal temporal grid (`8k + 1`), so the realized duration is quantized — about 0.33s per step at 24 fps — and it shifts with `frame_rate`, since the head predicts seconds rather than frames. `min_seconds` must be strictly less than `max_seconds`. These bounds are ignored when `num_frames` is set explicitly.

Bounds narrower than one grid step may not be satisfiable exactly: at 24 fps `[1.0s, 1.02s]` converts to `[24, 24]` frames, and 24 is not `8k + 1`. The nearest grid point is used and a warning is logged, so the returned length can fall just outside bounds that tight.

To inspect a prediction without generating a video, call the head directly. Everything it needs is public:

```py
prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False)
video_tokens, audio_tokens, _ = pipe.connectors(prompt_embeds, prompt_attention_mask)

num_frames = pipe.duration_head.predict_num_frames(
    video_tokens,
    audio_tokens,
    frame_rate=24.0,
    temporal_compression_ratio=pipe.vae_temporal_compression_ratio,
)
seconds = pipe.duration_head(video_tokens, audio_tokens).item()  # raw, before clamping
print(f"predicted {seconds:.2f}s -> {num_frames} frames")
```

Converting a 2.5 checkpoint picks the head up automatically with `--full_pipeline`, or on its own with `--duration_head`. Checkpoints predating 2.5 have no such weights, and conversion skips the component rather than failing.

### LTX-2.5 Modular

LTX-2.5 is also available as a modular pipeline. The default blockset uses the diffusion decoder and predicts the video duration when `num_frames` is omitted. It applies guidance separately to video and audio through the `guider` and `audio_guider` components. See [`LTX2Guidance`] for the available guidance parameters. By default, the modular pipeline will download the prompt enhancer and processor from the [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) repo. Below is a T2V modular example:

```py
import torch
from diffusers import ModularPipeline, ComponentsManager
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video

device = "cuda"
frame_rate = 24.0
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)

model_path = "Lightricks/LTX-2.5-Diffusers"

cm = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_path, components_manager=cm)
pipe.load_components(dtype=torch.bfloat16)
# Set memory_reserve_margin higher to more aggressively offload component models
cm.enable_auto_cpu_offload(device=device, memory_reserve_margin="20GB")
# The NATTEN processor works if `kernels` is available (`pip install kernels`)
# Otherwise omit the below line to use the Flex Attention processor
pipe.diffusion_decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
pipe.diffusion_decoder.enable_tiling()

prompt = (
    "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."
)

output_state = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=None,  # Set to an int (e.g. 121) to specify a fixed video length
    frame_rate=frame_rate,
    num_inference_steps=30,
    use_cross_timestep=True,
    enable_prompt_enhancement=True,
    generator=generator,
    output_type="np",
)
video = output_state.get("videos")
audio = output_state.get("audio")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_modular_t2v.mp4",
)
```

The modular pipeline will automatically switch workflows based on the supplied inputs. For example, if `image` is supplied, an I2V workflow will be used:

```py
import torch
from diffusers import ModularPipeline, ComponentsManager
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video, load_image

device = "cuda"
frame_rate = 24.0
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)

model_path = "Lightricks/LTX-2.5-Diffusers"

cm = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_path, components_manager=cm)
pipe.load_components(dtype=torch.bfloat16)
cm.enable_auto_cpu_offload(device=device, memory_reserve_margin="20GB")
pipe.diffusion_decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
pipe.diffusion_decoder.enable_tiling()

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion."
)
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = load_image(image_path)

output_state = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=None,  # Set to an int (e.g. 121) to specify a fixed video length
    frame_rate=frame_rate,
    num_inference_steps=30,
    use_cross_timestep=True,
    enable_prompt_enhancement=True,
    generator=generator,
    output_type="np",
)
video = output_state.get("videos")
audio = output_state.get("audio")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_modular_i2v.mp4",
)
```

#### Diffusion Fidelity Rendering (DFR)

`LTX2DFRBlocks` trades wall-clock time for detail fidelity. It generates on a canvas padded to a whole number of keyframe segments and spends one extra latent frame of tokens per segment border on a **keyframe slot** — a single-pixel-frame latent the model fills in. Relaxing the effective temporal compression at those positions means the surrounding video is conditioned on genuinely new frames rather than interpolated ones. This needs a transformer whose config sets `use_keyframes_abs_pos_embedding`, which LTX-2.5 checkpoints ship.

The recipe is two passes of the same blocks: a base pass at half resolution, then a detailing pass at full resolution seeded from it. Both the video latents and the keyframe slots are upsampled in between, and the spatial detailing IC-LoRA is loaded for the second pass only — so it goes on between the two calls, like the [stage 2 distilled LoRA](#stage-2-with-the-distilled-lora).

```py
import torch
from diffusers import ComponentsManager
from diffusers.modular_pipelines import LTX2DFRBlocks
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

device = "cuda"
model_path = "Lightricks/LTX-2.5-Diffusers"
prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn."
height, width, num_frames, frame_rate = 1024, 1536, 121, 24.0

cm = ComponentsManager()
pipe = LTX2DFRBlocks().init_pipeline(model_path, components_manager=cm)
pipe.load_components(dtype=torch.bfloat16)
cm.enable_auto_cpu_offload(device=device, memory_reserve_margin="20GB")

common = dict(prompt=prompt, num_frames=num_frames, frame_rate=frame_rate, output_type="latent")

# Pass 1: half resolution. Returns the video latents and one keyframe slot per segment border.
first = pipe(height=height // 2, width=width // 2, **common)
video_latents, keyframes_latents = first.get("videos"), first.get("keyframes_latents")

# Upsample both. `latents_normalized=False`: `output_type="latent"` already applied the latent statistics.
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path, subfolder="latent_upsampler", dtype=torch.bfloat16
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)


def upsample(latents):
    return upsample_pipe(
        latents=latents, latents_normalized=False, output_type="latent", return_dict=False
    )[0]


# Pass 2: full resolution, with the detailing IC-LoRA and the half-resolution result as its in-context
# reference. The adapter is calibrated for strength 0.5.
pipe.load_lora_weights("Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler", adapter_name="detailing")
pipe.set_adapters(["detailing"], weights=[0.5])

output = pipe(
    height=height,
    width=width,
    latents=upsample(video_latents),
    keyframes_latents=upsample(keyframes_latents),
    detailing_reference_latents=video_latents,
    detailing_reference_downscale_factor=2,
    **{**common, "output_type": "pt"},
)
video, audio = output.get("videos"), output.get("audio")
```

`height` and `width` are the output resolution and must be divisible by twice the VAE's spatial compression ratio (64 for LTX-2.5), since the base pass runs at half of each axis. Whatever `num_frames` asks for, the canvas is padded onto the segment grid internally and trimmed back before decoding, so the caller always gets the frame count it requested.

You can see the supported workflows in the docs for each blockset (e.g. [`LTX2AutoBlocks`], [`LTX25AutoBlocks`]).

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

## LTX2VideoDiffusionDecodePipeline

[[autodoc]] LTX2VideoDiffusionDecodePipeline
  - all
  - __call__

## LTX2DurationHead

[[autodoc]] pipelines.ltx2.duration_head.LTX2DurationHead
    - forward
    - predict_num_frames

## LTX2PipelineOutput

[[autodoc]] pipelines.ltx2.pipeline_output.LTX2PipelineOutput

## LTX2ModularPipeline

[[autodoc]] LTX2ModularPipeline

## LTX2AutoBlocks

[[autodoc]] LTX2AutoBlocks

## LTX25ModularPipeline

[[autodoc]] LTX25ModularPipeline

## LTX25AutoBlocks

[[autodoc]] LTX25AutoBlocks

## LTX2DFRModularPipeline

[[autodoc]] LTX2DFRModularPipeline

## LTX2DFRBlocks

[[autodoc]] LTX2DFRBlocks

## LTX2Guidance

[[autodoc]] modular_pipelines.ltx2.guider.LTX2Guidance
