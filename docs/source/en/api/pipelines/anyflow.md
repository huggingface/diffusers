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

[AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://huggingface.co/papers/<arxiv-id>) by Yuchao Gu et al.

*Few-step video generation has been significantly advanced by consistency models. However, their performance often degrades in any-step video diffusion models due to the fixed-point formulation. To address this limitation, we present AnyFlow, the first any-step video diffusion distillation framework built on flow maps. Instead of learning only the mapping z_t → z_0, AnyFlow learns transitions z_t → z_r over arbitrary time intervals, enabling a single model to adapt to different inference budgets. We design an improved forward flow map training recipe that fine-tunes pretrained video diffusion models into flow map models, and introduce Flow Map Backward Simulation to enable on-policy distillation for flow map models. Extensive experiments across both bidirectional and causal architectures, at scales ranging from 1.3B to 14B, on text-to-video and image-to-video tasks demonstrate that AnyFlow outperforms consistency-based baselines while preserving high fidelity and flexible sampling under varying step budgets.*

The original code can be found at [Enderfga/AnyFlow](https://github.com/Enderfga/AnyFlow). The project page is at [anyflow.github.io](https://anyflow.github.io/).

The following AnyFlow checkpoints are supported:

| Checkpoint | Backbone | Description |
|------------|----------|-------------|
| [`nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers) | Wan2.1 1.3B | Bidirectional T2V, lightweight |
| [`nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers) | Wan2.1 14B | Bidirectional T2V, full quality |
| [`nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers) | FAR + Wan2.1 1.3B | Causal T2V / I2V / TV2V |
| [`nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers`](https://huggingface.co/nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers) | FAR + Wan2.1 14B | Causal T2V / I2V / TV2V |

> [!TIP]
> Choose `AnyFlowPipeline` for traditional bidirectional text-to-video generation. Choose `AnyFlowFARPipeline` for streaming I2V, video continuation (TV2V), or any setup that benefits from frame-by-frame autoregressive sampling.

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

The causal pipeline selects between T2V / I2V / TV2V via the ``context_sequence`` argument: pass ``None``
for plain text-to-video, or a dict with a ``"raw"`` key holding a video tensor of shape
``(B, C, T, H, W)`` with ``T = 4n + 1`` to condition on existing frames. Use a single conditioning frame
for I2V and a longer clip for TV2V continuation.

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
    num_frames=33,
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
<hfoption id="i2v">

```py
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_image
from torchvision import transforms

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Wrap the conditioning image as a one-frame video tensor: (1, 3, 1, H, W)
first_frame = load_image("path/to/first_frame.png")
to_tensor = transforms.Compose([transforms.Resize((480, 832)), transforms.ToTensor()])
first_frame = to_tensor(first_frame).unsqueeze(0).unsqueeze(2).to("cuda")  # (1, 3, 1, 480, 832)

video = pipe(
    prompt="a cat walks across a sunlit lawn",
    context_sequence={"raw": first_frame},
    num_inference_steps=4,
    num_frames=33,
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
<hfoption id="tv2v">

```py
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video, load_video
from torchvision import transforms

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Provide a context clip whose frame count is 4n + 1 (e.g., 9, 13, 17).
context_frames = load_video("path/to/context.mp4")  # list of PIL frames
to_tensor = transforms.Compose([transforms.Resize((480, 832)), transforms.ToTensor()])
context_tensor = torch.stack([to_tensor(f) for f in context_frames[:9]], dim=1).unsqueeze(0).to("cuda")
# Shape: (1, 3, 9, 480, 832)

video = pipe(
    prompt="continue the story",
    context_sequence={"raw": context_tensor},
    num_inference_steps=4,
    num_frames=33,
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

</hfoption>
</hfoptions>

## Notes

- The released NVIDIA checkpoints went through a two-stage LoRA distillation: forward Flow-Map training plus on-policy distillation that combines Flow-Map backward simulation with **DMD reverse-divergence supervision** over the student's own rollouts. CFG was fused into the model weights during stage 1 (`fuse_guidance_scale = 3.0`), so inference does not run a second classifier-free guidance pass — quality is recovered from the distilled weights themselves.
- `FlowMapEulerDiscreteScheduler` is general-purpose. You can attach it to any flow-map-distilled checkpoint via `from_pretrained(..., scheduler=FlowMapEulerDiscreteScheduler.from_config(...))`.
- The bidirectional pipeline accepts any `AnyFlowTransformer3DModel` configured with `init_flowmap_model=True`. The causal pipeline additionally requires `init_far_model=True`.
- LoRA training is supported via `WanLoraLoaderMixin`, the same mixin used by the upstream Wan pipelines.
- For continued on-policy fine-tuning with DMD, both pipelines expose a `training_rollout` method that drives the three-segment Flow-Map backward simulation used in the original AnyFlow stage-2 trainer.

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
