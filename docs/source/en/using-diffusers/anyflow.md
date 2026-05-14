<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
the License for the specific language governing permissions and limitations under the License.
-->

# AnyFlow

[AnyFlow](https://huggingface.co/papers/2605.13724) is a video diffusion **distillation** framework that turns
a pretrained Wan2.1 teacher into an *any-step* student under standard Euler sampling. A single distilled
checkpoint can be evaluated at 1, 2, 4, 8, 16... NFE without retraining and quality scales **monotonically**
with steps — unlike consistency models, which often degrade as NFE grows.

The key idea is to learn the **flow map** $\Phi_{r\leftarrow t}: \mathbf{z}_t \to \mathbf{z}_r$ for arbitrary
$1 \ge t \ge r \ge 0$ instead of the fixed endpoint map $\mathbf{z}_t \to \mathbf{z}_0$ used by consistency
models. Composability of the flow map removes re-noising between sampling steps; on-policy distillation with
**DMD reverse-divergence supervision** plus **Flow-Map backward simulation** (3-segment shortcut) closes the
exposure-bias gap that consistency-based distillation leaves open.

AnyFlow was developed by Yuchao Gu, Guian Fang and collaborators at [NUS ShowLab](https://sites.google.com/view/showlab) in collaboration with NVIDIA. The original training code lives at [`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow); the project page is at [nvlabs.github.io/AnyFlow](https://nvlabs.github.io/AnyFlow). The four released checkpoints are grouped under the [`nvidia/anyflow`](https://huggingface.co/collections/nvidia/anyflow) Hugging Face collection.

This guide walks through the practical decisions: which pipeline to pick, how to use any-step sampling, and
how to plug AnyFlow into typical T2V / I2V / V2V workflows.

## Bidirectional vs causal — pick a pipeline

AnyFlow ships in two flavors that share the same scheduler and the same flow-map distillation but differ in
how they sample frames:

- [`AnyFlowPipeline`](../api/pipelines/anyflow#anyflowpipeline) — **bidirectional** T2V. Denoises the entire
  video tensor in one pass with global self-attention. Use this when the input is a single text prompt and you
  do not need streaming output.
- [`AnyFlowFARPipeline`](../api/pipelines/anyflow#anyflowfarpipeline) — **causal (FAR)**. Denoises the
  video chunk by chunk with block-sparse causal attention and reuses KV cache across chunks. Use this for
  image-to-video (I2V), video-to-video (V2V) continuation, or any setup that benefits from frame-level
  autoregressive sampling. The same model handles all three task modes via the `context_sequence` argument.

A quick selector:

| Scenario | Pipeline | How to invoke |
|----------|----------|---------------|
| Pure text-to-video, max quality at fixed NFE | `AnyFlowPipeline` | `pipe(prompt, ...)` |
| Image-to-video (start from a still image) | `AnyFlowFARPipeline` | `pipe(prompt, context_sequence={"raw": <one-frame tensor>}, ...)` |
| Video continuation / V2V | `AnyFlowFARPipeline` | `pipe(prompt, context_sequence={"raw": <multi-frame tensor>}, ...)` |
| Streaming / progressive generation | `AnyFlowFARPipeline` | — |

The bidirectional variant is faster per token at high resolution; the causal variant trades that for the
ability to start sampling before all latent frames are allocated, useful for very long sequences.

## Loading checkpoints

NVIDIA released four AnyFlow checkpoints, one per pipeline + scale combination:

```py
import torch
from diffusers import AnyFlowPipeline, AnyFlowFARPipeline

# Bidirectional, lightweight
pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Bidirectional, full quality
pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Causal (FAR), 1.3B
pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Causal (FAR), 14B
pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")
```

All four use the same [`FlowMapEulerDiscreteScheduler`](../api/schedulers/flow_map_euler_discrete) with
`shift=5.0` baked in.

## Any-step sampling

The defining feature of AnyFlow is that the same checkpoint produces increasing quality as you raise NFE,
with no schedule retuning. Sweep step counts on a fixed prompt to see how the model trades latency for
fidelity:

```py
import torch
from diffusers import AnyFlowPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A red panda eating bamboo in a forest, cinematic lighting"

for nfe in [1, 2, 4, 8, 16, 32]:
    # Re-seed the generator inside the loop so the only changing variable across runs is NFE.
    generator = torch.Generator("cuda").manual_seed(0)
    video = pipe(prompt, num_inference_steps=nfe, num_frames=33, generator=generator).frames[0]
    export_to_video(video, f"out_nfe{nfe}.mp4", fps=16)
```

In our benchmarks (paper Tab 3 / Fig 1) every AnyFlow checkpoint improves monotonically from 4 → 32 NFE
on VBench Quality, while consistency-based baselines (rCM, Self-Forcing) degrade in the same regime.

> [!TIP]
> Classifier-free guidance (CFG) was *fused* into the model weights during distillation
> (`fuse_guidance_scale = 3.0`). The pipeline does not run a second guided forward pass at inference time —
> guidance comes from the distilled weights themselves. Leave `guidance_scale=1.0` (the default) for the
> released checkpoints.

## Image-to-video and video-to-video

The causal pipeline supports three task modes from a single distilled model. The mode is selected
implicitly by the ``context_sequence`` argument (a dict with a ``"raw"`` video tensor or ``"latent"``
pre-encoded latents). Frame counts in the context tensor must satisfy ``T = 4n + 1`` to align with the
VAE temporal stride.

> [!IMPORTANT]
> The FAR pipeline runs a chunked rollout, and `num_frames` must agree with the chunk schedule. The default
> `chunk_partition=[1, 3, 3, 3, 3, 3, 3, 2]` sums to 21 latent frames, which is matched to the released
> checkpoints' canonical `num_frames=81` (21 = (81 − 1) // 4 + 1). When you change `num_frames`, you **must**
> pass a matching `chunk_partition` whose entries sum to `(num_frames - 1) // 4 + 1`, otherwise the pipeline
> raises an `AssertionError`. For example, `num_frames=33` corresponds to 9 latent frames, so a valid
> override is `chunk_partition=[1, 4, 4]`.

```py
import numpy as np
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")


def to_video_tensor(images, height=480, width=832):
    """Convert a list of PIL images into the (B, C, T, H, W) [0, 1] tensor the FAR pipeline consumes."""
    frames = np.stack([np.asarray(img.resize((width, height))) for img in images]).astype("float32") / 255.0
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)


# 1) Text-to-video (no context). 81 frames matches the default chunk_partition.
video = pipe(prompt="A cat surfing a wave at sunset", num_inference_steps=4, num_frames=81).frames[0]
export_to_video(video, "t2v.mp4", fps=16)

# 2) Image-to-video — a single conditioning frame produces 1 latent frame, which exactly fits the first
# entry of the default chunk_partition (`[1, 3, 3, ...]`).
first_frame = load_image("path/to/first_frame.png")
context_tensor = to_video_tensor([first_frame]).to("cuda")  # (1, 3, 1, 480, 832), [0, 1]
video = pipe(
    prompt="a cat walks across a sunlit lawn",
    context_sequence={"raw": context_tensor},
    num_inference_steps=4,
    num_frames=81,
).frames[0]
export_to_video(video, "i2v.mp4", fps=16)

# 3) Video-to-video continuation. A 9-frame raw context maps to 3 latent frames; we override
# chunk_partition so the first chunk covers the whole context exactly.
context_frames = load_video("path/to/context.mp4")[:9]  # 9 = 4·2 + 1
context_tensor = to_video_tensor(context_frames).to("cuda")  # (1, 3, 9, 480, 832)
video = pipe(
    prompt="continue the story",
    context_sequence={"raw": context_tensor},
    num_inference_steps=4,
    num_frames=81,
    chunk_partition=[3, 3, 3, 3, 3, 3, 3],  # 7 chunks × 3 = 21 latent frames; first chunk = context
).frames[0]
export_to_video(video, "v2v.mp4", fps=16)
```

Internally, the patchification chunk schedule depends on whether (and how long) ``context_sequence`` is set:
without context the model uses kernel sizes 2 (full) and 4 (compressed); with a context clip the first chunk
uses kernel size 1 so the conditioning frames keep full resolution.

If you already have VAE-encoded latents, pass them via ``context_sequence={"latent": ...}`` to skip the
``vae_encode`` step.

## Memory and inference speed

A 14B AnyFlow model fits on a single 40 GB device with group offloading + VAE slicing:

```py
import torch
from diffusers import AnyFlowPipeline
from diffusers.hooks import apply_group_offloading

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
)
apply_group_offloading(pipe.transformer, onload_device="cuda", offload_type="leaf_level")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

For latency, `torch.compile` works well on the transformer (the heaviest module by far):

```py
pipe = pipe.to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
```

Compile costs are amortized after a few steps; combined with low NFE (4–8 for AnyFlow) `torch.compile`
delivers a noticeable speedup over eager mode on the 14B path.

## LoRA fine-tuning

Both pipelines reuse [`WanLoraLoaderMixin`](../api/loaders/lora), so any LoRA adapter trained for the
matching Wan2.1 backbone loads directly:

```py
pipe.load_lora_weights("path/or/repo/with/wan_lora")
```

For continued **on-policy** fine-tuning with DMD-style reverse-divergence supervision (the same recipe used
to produce the released checkpoints), refer to the original AnyFlow training framework at
[`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow), which is out of scope for diffusers.

## Common gotchas

- **Always-1.0 `guidance_scale`.** The distilled checkpoints already encode CFG. Setting `guidance_scale > 1`
  will run a redundant unconditional pass, double the latency, and slightly hurt quality.
- **Bidirectional pipeline does not stream.** All `num_frames` worth of latents are denoised together. Use
  the causal pipeline if you want to start playback before sampling completes.
- **Causal pipeline KV cache assumes the chunk schedule is consistent across calls.** Rebuilding the cache
  mid-generation is not supported by the released model.
- **`num_frames` must satisfy the VAE temporal stride.** Use values of the form `(N - 1) % 4 == 0` (e.g., 9,
  17, 33, 81) for the released checkpoints.

## Citation

```bibtex
@misc{gu2026anyflowanystepvideodiffusion,
  title={AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation},
  author={Yuchao Gu and Guian Fang and Yuxin Jiang and Weijia Mao and Song Han and Han Cai and Mike Zheng Shou},
  year={2026},
  eprint={2605.13724},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.13724},
}

@article{gu2025long,
  title={Long-Context Autoregressive Video Modeling with Next-Frame Prediction},
  author={Gu, Yuchao and Mao, Weijia and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.19325},
  year={2025}
}
```
