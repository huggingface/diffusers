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
2. **Stage 2 — LTX-2 refiner (optional).** A separate sink-bidirectional Euler refiner pipeline
   ([`SanaWMLTX2Refiner`]) that wraps
   [`SanaWMLTX2RefinerTransformer3DModel`] + `LTX2TextConnectors` and a Gemma-3 text encoder, run for 3
   distilled sigma steps.

Both stages decode through the [`AutoencoderKLLTX2Video`] VAE.

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/SANA-WM_bidirectional-diffusers`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers) | `torch.bfloat16` |
| [`Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner) | `torch.bfloat16` |

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
)
pipe.enable_model_cpu_offload()  # ~45 GB of weights — offload between stages

# SANA-WM was trained on the LTX-2 VAE in framewise mode with tiling enabled. Without these
# settings the VAE encodes the whole (B, C, T, H, W) clip in one shot, which gives subtly
# different numerics from the released checkpoint.
pipe.vae.enable_tiling()
pipe.vae.use_framewise_encoding = True
pipe.vae.use_framewise_decoding = True
pipe.vae.tile_sample_stride_num_frames = 64
pipe.vae.tile_sample_min_num_frames = 96

prompt = "A car driving across a vast desert plain at golden hour."
output = pipe(
    image=Image.open("input.png").convert("RGB"),
    prompt=prompt,
    action="w-80,jw-40,w-40",        # WASD-style action DSL: forward 80f, jump+forward 40f, forward 40f
    intrinsics=[800.0, 800.0, 845.0, 464.0],  # fx, fy, cx, cy in original-image pixels
    num_frames=161,
    num_inference_steps=60,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="latent",            # hand the latents to the refiner below
)
```

Pass `action=None` and supply your own `c2w` poses (`(F, 4, 4)` numpy array) to drive the camera trajectory
explicitly. Drop `output_type="latent"` to get video straight out of stage 1 and skip the refiner.

### Stage 2 — the LTX-2 refiner

[`SanaWMLTX2Refiner`] ships as its own repository, the way SDXL splits base and refiner. Pass the base pipeline's
VAE so the weights are shared rather than loaded twice:

```python
from diffusers import SanaWMLTX2Refiner

refiner = SanaWMLTX2Refiner.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner",
    vae=pipe.vae,
    torch_dtype=torch.bfloat16,
)
refiner.enable_model_cpu_offload()

frames = refiner(output.latent, prompt, fps=16)
export_to_video(list(frames), "sana_wm.mp4", fps=16)
```

Without a `vae` the refiner returns refined latents instead of video, which is useful if you want to decode
yourself.

> [!TIP]
> Enable offloading on **both** pipelines, or free stage 1 before stage 2 (`pipe.transformer.to("cpu")`). The two
> stages together are around 45 GB in `torch.bfloat16`, and keeping both resident on one 80 GB card leaves too
> little room for activations. Note also that stage 2 honours `torch_dtype`: loading the refiner in
> `torch.float32` roughly doubles its memory and changes the output slightly.

If you don't have camera intrinsics, [`pi3-vision`](https://github.com/OliverSFAC/pi3-vision) can estimate them
from a single frame:

```python
from diffusers.pipelines.sana_wm.cam_utils import estimate_intrinsics_with_pi3x
intrinsics = estimate_intrinsics_with_pi3x(image)  # `pip install pi3-vision`
```

## Converting the released checkpoint

If you have the source SANA-WM release (not the pre-converted diffusers snapshot), run the conversion script once:

```bash
python scripts/convert_sana_wm_to_diffusers.py \
    --src Efficient-Large-Model/SANA-WM_bidirectional \
    --dst ./SANA-WM_bidirectional-diffusers
```

This writes two directories: the base pipeline at `--dst`, and the stage-2 refiner alongside it at
`./SANA-WM_bidirectional-diffusers-refiner` (override with `--dst-refiner`). Then load each from its local path
as usual.

## Components

- `tokenizer` — [`GemmaTokenizerFast`]
- `text_encoder` — Gemma-2 (returns decoder hidden states)
- `vae` — [`AutoencoderKLLTX2Video`] (LTX-2, spatial ×32 / temporal ×8)
- `transformer` — [`SanaWMTransformer3DModel`], 1.6B-parameter bidirectional DiT
- `scheduler` — [`FlowMatchEulerDiscreteScheduler`]

The stage-2 refiner is a separate repository with its own `transformer`
([`SanaWMLTX2RefinerTransformer3DModel`]), `connectors` (`LTX2TextConnectors`), `tokenizer`, Gemma-3
`text_encoder` and `scheduler`. It has no `vae` of its own — pass the base pipeline's.

## SanaWMPipeline

[[autodoc]] SanaWMPipeline
  - all
  - __call__

## SanaWMLTX2Refiner

The LTX-2 stage-2 refiner is a standalone [`DiffusionPipeline`] that takes stage-1 latents. Give it a `vae` (the
base pipeline's, so the weights are shared) to have it decode to video; without one it returns refined latents.

[[autodoc]] SanaWMLTX2Refiner
  - all
  - __call__

## SanaWMPipelineOutput

[[autodoc]] pipelines.sana_wm.pipeline_output.SanaWMPipelineOutput
