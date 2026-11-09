<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLMiniMaxH3Image

The image variational autoencoder (VAE) model with KL loss used in [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax. It pairs the causal 3D CNN encoder of [`AutoencoderKLMiniMaxH3`], run over a single frame, with a 2D ViT decoder, and compresses 16x spatially. `encode` and `decode` take and return 4D image tensors.

It shares three conventions with [`AutoencoderKLMiniMaxH3`]:

- **Latents are normalized per channel.** There is no `scaling_factor`: a pipeline encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.
- **The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range**, not the usual `[-1, 1]`. `encode` expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline applies `sample * imagenet_std + imagenet_mean` and clamps to `[0, 1]` before postprocessing.
- **Spatial tiling is on by default.** MiniMax-H3-Image was released with tiling enabled for both encoding and decoding and the released images are the blended-tile ones, so turning it off changes the output. Use `enable_tiling` to change the tile geometry and `disable_tiling` to switch it off.

```python
import torch
from diffusers import AutoencoderKLMiniMaxH3Image

vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
    "MiniMaxAI/MiniMax-H3-Image", subfolder="vae", dtype=torch.float32
).to("cuda")  # or "mps", "xpu", "cpu"
```

## AutoencoderKLMiniMaxH3Image

[[autodoc]] AutoencoderKLMiniMaxH3Image
    - encode
    - decode
    - all
