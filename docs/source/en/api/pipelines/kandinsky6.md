<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 6

Kandinsky 6 is a family of video generation models from [Kandinsky Lab](https://huggingface.co/kandinskylab). The
main model generates video and synchronized audio from text or a reference image with a single multimodal diffusion
transformer: video and audio latents are denoised together through fused blocks that cross-attend between the two
modalities, each conditioned on its own Qwen2.5-VL text branch and a CLIP pooled embedding. A separate
super-resolution model upscales the generated video tile by tile in the latent space of a causal 3D K-VAE.

> [!TIP]
> Check out the [Kandinsky Lab](https://huggingface.co/kandinskylab) organization on the Hub for the full set of
> official checkpoints, including flow-matching and distilled variants of both the base and super-resolution models.
>
> Distilled checkpoints ship with the few-step [`PiflowScheduler`] and must be run with `guidance_scale=1.0`.

## Available models

| Model | Pipeline | Notes |
|---|---|---|
| [`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Distilled, `guidance_scale=1.0`, 10 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Distilled, `guidance_scale=1.0`, 10 steps |
| [`kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers) | [`Kandinsky6TI2VAPipeline`] | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers) | [`Kandinsky6SRPipeline`] | Flow matching super-resolution |
| [`kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) | [`Kandinsky6SRPipeline`] | Distilled super-resolution, 2 steps |

## Text/image-to-video-and-audio

```python
import torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

pipe = Kandinsky6TI2VAPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat and a dog baking a cake together in a kitchen.",
    height=480,
    width=864,
    num_frames=121,
    num_inference_steps=10,
    guidance_scale=1.0,
)
encode_video(
    output.frames[0],
    fps=24,
    output_path="output.mp4",
    audio=output.audio[0][None],
    audio_sample_rate=pipe.audio_sample_rate,
)
```

Pass `image=` to condition the first frame on a reference image, `sample_audio=False` to generate video only, and
`expand_prompts=True` to let the Qwen2.5-VL text encoder rewrite short prompts into detailed ones first.

## Video super-resolution

[`Kandinsky6SRPipeline`] takes the frames produced by [`Kandinsky6TI2VAPipeline`] and upscales them by `2`, `4`, or
`2.25` (a 1.125x bilinear pre-upscale followed by the 2x path). The video is split into overlapping tiles, every tile
is refined at one of the tile sizes the SR transformer was trained on, and the tiles are blended back with Hann
windows.

```python
# required: lets inductor pick flex-attention tiles that fit the SR block mask
torch._inductor.config.max_autotune = True  

sr_pipe = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", torch_dtype=torch.bfloat16
)
# The SR transformer always runs NABLA sparse attention on the `flex` backend. Compile it, otherwise flex falls
# back to an eager implementation that needs far more memory at video resolutions.
sr_pipe.enable_model_cpu_offload()
sr_pipe.transformer.set_attention_backend("flex")
sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)

upscaled = sr_pipe(video=output.frames[0], resolution_scale=2.25, num_inference_steps=2).frames[0]
```

## Memory optimization

Refer to the [Reduce memory usage](../../optimization/memory) guide for the general set of techniques. Both
[`Kandinsky6TI2VAPipeline`] and [`Kandinsky6SRPipeline`] support [model offloading](../../optimization/memory#model-offloading)
(used above) and, for a smaller footprint at the cost of speed, [sequential CPU offloading](../../optimization/memory#cpu-offloading):

```python
pipe.enable_sequential_cpu_offload()
```

[`Kandinsky6TI2VAPipeline`]'s video VAE also supports [tiled decoding](../../optimization/memory#vae-tiling) for high
resolutions or long videos:

```python
pipe.vae.enable_tiling()
```

## Notes

- `height` and `width` must be divisible by the video VAE's spatial compression ratio times the transformer's patch
  size — `16` with the default [`Kandinsky6TI2VAPipeline`] configuration (`AutoencoderKLHunyuanVideo` at a
  compression ratio of `8`, `patch_size=(1, 2, 2)`). `480x864`, used in the example above, satisfies this.
- [`Kandinsky6SRPipeline`]'s input `video` must have `1 + k * vae_scale_factor_temporal` frames for some integer `k`
  (a temporal compression ratio of `4` with the default K-VAE configuration, so `121` frames works but `120` doesn't)
  — trim or pad a video that doesn't already satisfy this before upscaling it.
- `expand_prompts=True` reuses the already-loaded Qwen2.5-VL text encoder for an extra generation pass before
  denoising, so it adds latency but no extra model weights.
- Compile the repeated transformer blocks for faster repeated inference:
  ```python
  pipe.transformer.compile_repeated_blocks(fullgraph=True)
  ```

## Kandinsky6TI2VAPipeline

[[autodoc]] Kandinsky6TI2VAPipeline
  - all
  - __call__

## Kandinsky6SRPipeline

[[autodoc]] Kandinsky6SRPipeline
  - all
  - __call__

## Kandinsky6SRLatentUpscalerBank

[[autodoc]] Kandinsky6SRLatentUpscalerBank
  - forward

## Kandinsky6TI2VAPipelineOutput

[[autodoc]] pipelines.kandinsky6.pipeline_output.Kandinsky6TI2VAPipelineOutput

## Kandinsky6SRPipelineOutput

[[autodoc]] pipelines.kandinsky6.pipeline_output.Kandinsky6SRPipelineOutput
