<!-- Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# SANA-WM

SANA-WM is a camera-controlled image-to-video world model built on top of SANA. Given a first-frame image, a text
prompt, and a camera trajectory (either explicit `c2w` poses or a WASD/IJKL action string), it generates a video
whose motion follows the requested camera path.

Inference runs in two stages:

1. **Stage 1 — SANA-WM DiT.** A 1.6B-parameter bidirectional DiT with GDN-Triton linear attention and a UCPE
   camera-control branch. Sampling uses an LTX-style flow-matching Euler scheduler with per-token timesteps; the
   first latent frame is the conditioning anchor.
2. **Stage 2 — LTX-2 refiner (optional).** A sink-bidirectional Euler refiner ([`SanaWMLTX2Refiner`]) that wraps
   diffusers' own `LTX2VideoTransformer3DModel` + `LTX2TextConnectors` and a Gemma-3 text encoder, run for 3
   distilled sigma steps.

Both stages decode through the [`AutoencoderKLLTX2Video`] VAE.

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/SANA-WM_bidirectional-diffusers`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers) | `torch.bfloat16` |

> [!TIP]
> SANA-WM is trained at a fixed 704×1280 resolution. The recommended dtype is for the transformer weights — keep
> the text encoder in `torch.bfloat16` and the VAE in `torch.float32` for best numerics. The pipeline expects
> camera intrinsics `[fx, fy, cx, cy]` in *original-image* pixel coordinates; the resize-and-center-crop transform
> is applied internally.

## Inference

```python
import torch
from PIL import Image

from diffusers import SanaWMPipeline
from diffusers.utils import export_to_video

pipe = SanaWMPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = Image.open("input.png").convert("RGB")

output = pipe(
    image=image,
    prompt="A car driving across a vast desert plain at golden hour.",
    action="w-80,jw-40,w-40",        # WASD-style action DSL: forward 80f, jump+forward 40f, forward 40f
    intrinsics=[800.0, 800.0, 845.0, 464.0],  # fx, fy, cx, cy in original-image pixels
    num_frames=161,
    num_inference_steps=60,
    guidance_scale=5.0,
    seed=42,
)
export_to_video(list(output.frames), "sana_wm.mp4", fps=16)
```

Pass `action=None` and supply your own `c2w` poses (`(F, 4, 4)` numpy array) to drive the camera trajectory
explicitly. Set `use_refiner=False` to skip stage 2.

## SanaWMPipeline

[[autodoc]] SanaWMPipeline
  - all
  - __call__

## SanaWMLTX2Refiner

[[autodoc]] SanaWMLTX2Refiner

## SanaWMPipelineOutput

[[autodoc]] pipelines.sana_wm.pipeline_output.SanaWMPipelineOutput
