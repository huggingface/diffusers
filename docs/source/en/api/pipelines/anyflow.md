<!-- Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <a href="https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_pipeline.py">
            <img alt="LoRA" src="https://img.shields.io/badge/LoRA-supported-green">
        </a>
    </div>
</div>

# AnyFlow

[AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://huggingface.co/papers/2605.13724) by Yuchao Gu, Guian Fang and collaborators at [NUS ShowLab](https://sites.google.com/view/showlab) in collaboration with NVIDIA.

*Few-step video generation has been significantly advanced by consistency models. However, their performance often degrades in any-step video diffusion models due to the fixed-point formulation. To address this limitation, we present AnyFlow, the first any-step video diffusion distillation framework built on flow maps. Instead of learning only the mapping z_t → z_0, AnyFlow learns transitions z_t → z_r over arbitrary time intervals, enabling a single model to adapt to different inference budgets. We design an improved forward flow map training recipe that fine-tunes pretrained video diffusion models into flow map models, and introduce Flow Map Backward Simulation to enable on-policy distillation for flow map models. Extensive experiments across both bidirectional and causal architectures, at scales ranging from 1.3B to 14B, on text-to-video and image-to-video tasks demonstrate that AnyFlow outperforms consistency-based baselines while preserving high fidelity and flexible sampling under varying step budgets.*

The original training code is at [`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow). The project page is at [nvlabs.github.io/AnyFlow](https://nvlabs.github.io/AnyFlow).

The following AnyFlow checkpoints are supported:

| Checkpoint | Backbone | Description |
|------------|----------|-------------|
| [`nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers) | Wan2.1 1.3B | Bidirectional T2V, lightweight |
| [`nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers) | Wan2.1 14B | Bidirectional T2V, full quality |
| [`nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers) | FAR + Wan2.1 1.3B | Causal T2V / I2V / V2V |
| [`nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers) | FAR + Wan2.1 14B | Causal T2V / I2V / V2V |

All four are grouped under the [`nvidia/anyflow`](https://huggingface.co/collections/nvidia/anyflow) Hugging Face collection.

> [!TIP]
> Choose `AnyFlowPipeline` for traditional bidirectional text-to-video generation. Choose `AnyFlowFARPipeline` for streaming I2V, video continuation (V2V), or any setup that benefits from frame-by-frame autoregressive sampling.

> [!TIP]
> AnyFlow supports any-step sampling: a single distilled checkpoint can be evaluated at 1, 2, 4, 8, 16... NFE without retraining. Quality scales monotonically with steps in our benchmarks.

### Optimizing Memory and Inference Speed

<hfoptions id="optimization">
<hfoption id="memory">

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

</hfoption>
<hfoption id="inference speed">

```py
import torch
from diffusers import AnyFlowPipeline

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
```

</hfoption>
</hfoptions>

### Generation with AnyFlow (Bidirectional T2V)

<hfoptions id="anyflow-bidi">
<hfoption id="usage">

```py
import torch
from diffusers import AnyFlowPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A red panda eating bamboo in a forest, cinematic lighting"
video = pipe(prompt, num_inference_steps=4, num_frames=33).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
</hfoptions>

### Generation with AnyFlow (FAR Causal)

The causal pipeline selects between T2V / I2V / V2V via the ``video`` (or ``video_latents``) argument:
omit both for plain text-to-video, or pass ``video=<tensor>`` of shape ``(B, T, C, H, W)`` in ``[0, 1]``
with ``T = 4n + 1`` to condition on existing frames. Use a single conditioning frame for I2V and a longer
clip for V2V continuation. If you already have pre-encoded latents in the model layout, pass them via
``video_latents=<tensor>`` to skip VAE encoding. ``video`` and ``video_latents`` are mutually exclusive.

> [!IMPORTANT]
> `AnyFlowFARPipeline.default_chunk_partition = [1, 3, 3, 3, 3, 3, 3, 2]` (sum 21) is matched to the
> released checkpoints' canonical 81 raw frames (21 latent frames at the VAE temporal stride of 4). When
> you change `num_frames`, you must also pass a matching `chunk_partition` summing to
> `(num_frames - 1) // 4 + 1`, otherwise the pipeline raises an `AssertionError`.

<hfoptions id="anyflow-far">
<hfoption id="t2v">

```py
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

video = pipe(
    prompt="A cat surfing a wave, sunset",
    num_inference_steps=4,
    num_frames=81,
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
<hfoption id="i2v">

```py
import numpy as np
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_image

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Wrap the conditioning image as a one-frame video tensor: (1, 1, 3, H, W) in [0, 1].
first_frame = load_image("path/to/first_frame.png").resize((832, 480))
arr = np.asarray(first_frame).astype("float32") / 255.0  # (480, 832, 3)
context_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to("cuda")

video = pipe(
    prompt="a cat walks across a sunlit lawn",
    video=context_tensor,
    num_inference_steps=4,
    num_frames=81,
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
<hfoption id="v2v">

```py
import numpy as np
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_video

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Context clip — 9 raw frames map to 3 latent frames (9 = 4·2 + 1, 3 = 2 + 1).
context_frames = load_video("path/to/context.mp4")[:9]
arr = np.stack([np.asarray(f.resize((832, 480))) for f in context_frames]).astype("float32") / 255.0
# np.stack gives (T, H, W, C) = (9, 480, 832, 3) → permute to (T, C, H, W) then add batch.
context_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to("cuda")  # (1, 9, 3, 480, 832)

video = pipe(
    prompt="continue the story",
    video=context_tensor,
    num_inference_steps=4,
    num_frames=81,
    # Override chunk_partition so the first chunk covers exactly the 3 latent context frames.
    chunk_partition=[3, 3, 3, 3, 3, 3, 3],
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
</hfoptions>

## Notes

- Classifier-free guidance is fused into the released checkpoints, so inference does not run a second guided forward pass. Keep the default `guidance_scale=1.0` unless your own checkpoint requires otherwise.
- `FlowMapEulerDiscreteScheduler` is general-purpose. You can attach it to any flow-map-distilled checkpoint via `from_pretrained(..., scheduler=FlowMapEulerDiscreteScheduler.from_config(...))`.
- `AnyFlowPipeline` uses [`AnyFlowTransformer3DModel`](../models/anyflow_transformer3d) (bidirectional). `AnyFlowFARPipeline` uses [`AnyFlowFARTransformer3DModel`](../models/anyflow_far_transformer3d), which adds a compressed-frame patch embedding and the FAR causal block-mask.
- LoRA loading is supported via `WanLoraLoaderMixin`, the same mixin used by the upstream Wan pipelines.
- For training recipes (forward flow-map training and on-policy distillation), refer to the original AnyFlow training framework at [`NVlabs/AnyFlow`](https://github.com/NVlabs/AnyFlow); training is out of scope for diffusers.

## AnyFlowPipeline

[[autodoc]] AnyFlowPipeline
  - all
  - __call__

## AnyFlowFARPipeline

[[autodoc]] AnyFlowFARPipeline
  - all
  - __call__

## AnyFlowPipelineOutput

[[autodoc]] pipelines.anyflow.pipeline_output.AnyFlowPipelineOutput
