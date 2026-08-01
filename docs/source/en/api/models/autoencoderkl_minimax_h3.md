<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLMiniMaxH3

The video variational autoencoder (VAE) model with KL loss used in [MiniMax-H3](https://huggingface.co/MiniMaxAI) by MiniMax. It pairs a causal 3D CNN encoder with a non-causal ViT decoder and compresses 16x spatially and 4x temporally.

Three things set it apart from most autoencoders in the library:

- **Latents are normalized per channel.** There is no `scaling_factor`: a pipeline encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.
- **The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range**, not the usual `[-1, 1]`. `encode` expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline applies `sample * imagenet_std + imagenet_mean` and clamps to `[0, 1]` before postprocessing.
- **Spatial tiling is on by default.** MiniMax-H3 was released with tiling enabled for both encoding and decoding and the released frames are the blended-tile ones, so turning it off changes the output. Use `enable_tiling` to change the tile geometry and `disable_tiling` to switch it off.

The temporal geometry is fixed by `clip_length` (17 pixel frames per encoder chunk) and `token_drop` (3 trailing latent frames dropped per encode), so `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.

```python
import torch
from diffusers import AutoencoderKLMiniMaxH3

vae = AutoencoderKLMiniMaxH3.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="vae", dtype=torch.float32
).to("cuda")
```

## AutoencoderKLMiniMaxH3

[[autodoc]] AutoencoderKLMiniMaxH3
    - encode
    - decode
    - all
