<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLive

[EdenLive](https://github.com/VAST-AI-Research/EdenLive-Official) generates video from a proxy video, a reference
image, and a text prompt. [`EdenLivePipeline`] supports bidirectional generation; [`EdenLiveCausalPipeline`]
generates a complete video in causal chunks and supports changing prompts between chunks.

## Basic usage

Prepare an RGB reference image and a proxy video with at least 49 frames sampled at 24 fps. The image provides
appearance, while the proxy provides motion and scene structure.

```python
import torch
from diffusers import EdenLivePipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = EdenLivePipeline.from_pretrained(
    "VAST-AI/EdenLive-5B-Diffusers",
    torch_dtype={"default": torch.bfloat16, "proxy_encoder": torch.float32, "proxy_projection": torch.float32},
).to("cuda")

image = load_image("reference.png")
video = load_video("proxy-24fps.mp4")

frames = pipe(
    image=image,
    video=video,
    prompt="A character explores a snowy landscape.",
    num_frames=49,
).frames[0]
export_to_video(frames, "edenlive.mp4", fps=24)
```

Keep the proxy encoder and projection in FP32 when loading the other models in BF16, as shown above.

## Causal generation

Load the causal transformer and scheduler from the same repository. This works with either model size and uses
four fixed denoising steps per chunk.

```python
from diffusers import EdenLiveCausalPipeline, EdenLiveTransformer3DModel, EdenLiveFlowMatchEulerDiscreteScheduler

causal_pipe = EdenLiveCausalPipeline.from_pretrained(
    "VAST-AI/EdenLive-5B-Diffusers",
    transformer=EdenLiveTransformer3DModel.from_pretrained(
        "VAST-AI/EdenLive-5B-Diffusers", subfolder="transformer_causal", torch_dtype=torch.bfloat16
    ),
    scheduler=EdenLiveFlowMatchEulerDiscreteScheduler.from_pretrained(
        "VAST-AI/EdenLive-5B-Diffusers", subfolder="scheduler_causal"
    ),
    torch_dtype={"default": torch.bfloat16, "proxy_encoder": torch.float32, "proxy_projection": torch.float32},
).to("cuda")
```

Use `prompt` for fixed text, or `chunk_prompts` to change it during a single video. At the default chunk size,
49 frames require five prompts. Reusing the image and video above:

```python
frames = causal_pipe(
    image=image,
    video=video,
    chunk_prompts=[
        "A character walks through snow.",
        "A character walks through snow.",
        "The character approaches a campfire.",
        "The character warms their hands beside the campfire.",
        "The character rests beside the campfire.",
    ],
    num_frames=49,
).frames[0]
export_to_video(frames, "chunk-prompts.mp4", fps=24)
```

`chunk_prompts` overrides a single `prompt` and cannot be combined with `prompt_embeds`. The negative prompt remains
fixed. A list passed as `prompt` represents a batch of independent videos. Each pipeline call starts a new video.

## LightTAE decoder

The full Wan VAE is the default. For faster decoding, pass the optional [`EdenLiveLightTAEDecoder`] to either pipeline:

```python
from diffusers import EdenLiveLightTAEDecoder

pipe = EdenLivePipeline.from_pretrained(
    "VAST-AI/EdenLive-5B-Diffusers",
    decoder=EdenLiveLightTAEDecoder.from_pretrained(
        "VAST-AI/EdenLive-5B-Diffusers", subfolder="decoder", torch_dtype=torch.bfloat16
    ),
    torch_dtype={"default": torch.bfloat16, "proxy_encoder": torch.float32, "proxy_projection": torch.float32},
).to("cuda")
```

## Notes

| Checkpoint | Default height × width | Example fps | Bidirectional steps |
|---|---|---|---|
| [5B](https://huggingface.co/VAST-AI/EdenLive-5B-Diffusers) | 704 × 1280 | 24 | 48 |
| [14B](https://huggingface.co/VAST-AI/EdenLive-14B-Diffusers) | 480 × 832 | 16 | 30 |

- Sample proxy frames at the desired output frame rate. `num_frames` must be $4k+1$ and fit within the input video;
  omitting it uses the largest valid prefix. Custom dimensions must be multiples of 32 for 5B or 16 for 14B.
- Guidance defaults to 3 for bidirectional generation and 1 for causal generation. Set `guidance_scale` above one
  to use a `negative_prompt`.
- For CPU offload, load without `.to("cuda")` and call `enable_model_cpu_offload()`.
- See the [example README](https://github.com/huggingface/diffusers/tree/main/examples/edenlive) for installation,
  video resampling, multi-GPU inference, and checkpoint conversion.

## EdenLivePipeline

[[autodoc]] EdenLivePipeline
    - all
    - __call__

## EdenLiveCausalPipeline

[[autodoc]] EdenLiveCausalPipeline
    - all
    - __call__

## EdenLiveLightTAEDecoder

[[autodoc]] EdenLiveLightTAEDecoder
    - decode

## EdenLivePipelineOutput

[[autodoc]] pipelines.edenlive.pipeline_output.EdenLivePipelineOutput
