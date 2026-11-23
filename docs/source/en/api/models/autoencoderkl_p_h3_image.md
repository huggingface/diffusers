<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLPH3Image

The image VAE of [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) by Pruna AI. It uses the encoder and latent space of MiniMax-H3's [video VAE](./autoencoderkl_minimax_h3), with a ViT decoder that decodes a single latent frame into an image.

It follows the conventions of [`AutoencoderKLMiniMaxH3`]:

- Images are passed as single-frame videos, `(batch_size, channels, 1, height, width)`, with 16x spatial compression.
- Latents are normalized per channel with `latents_mean` / `latents_std` instead of a `scaling_factor`.
- Pixels are ImageNet-normalized RGB over `[0, 1]`: `encode` expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns that space.
- Spatial tiling is on by default, with 256 pixel tiles and at least 64 pixels of overlap.

```python
import torch
from diffusers import AutoencoderKLPH3Image

vae = AutoencoderKLPH3Image.from_pretrained("PrunaAI/P-H3-Image", subfolder="vae", dtype=torch.bfloat16).to(
    "cuda"
)  # or "mps", "xpu", "cpu"
```

The encoder and the decoder's output projection stay in float32 whatever `dtype` the model is loaded in.

## AutoencoderKLPH3Image

[[autodoc]] AutoencoderKLPH3Image
    - encode
    - decode
    - all
