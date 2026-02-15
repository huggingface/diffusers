<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderRAE

`AutoencoderRAE` is a representation autoencoder that combines a frozen vision encoder (DINOv2, SigLIP2, or MAE) with a ViT-MAE-style decoder.

Paper: [Diffusion Transformers with Representation Autoencoders](https://huggingface.co/papers/2510.11690).

The model follows the standard diffusers autoencoder API:
- `encode(...)` returns an `EncoderOutput` with a `latent` tensor.
- `decode(...)` returns a `DecoderOutput` with a `sample` tensor.

## Usage

```python
import torch
from diffusers import AutoencoderRAE

model = AutoencoderRAE(
    encoder_cls="dinov2",
    encoder_name_or_path="facebook/dinov2-with-registers-base",
    encoder_input_size=224,
    patch_size=16,
    image_size=256,
).to("cuda").eval()

# Encode and decode
x = torch.randn(1, 3, 256, 256, device="cuda")
with torch.no_grad():
    latents = model.encode(x).latent
    recon = model.decode(latents).sample
```

`encoder_cls` supports `"dinov2"`, `"siglip2"`, and `"mae"`.

For latent normalization, use `latents_mean` and `latents_std` (matching other diffusers autoencoders).

For training, `forward(...)` also supports:
- `return_loss=True`
- `reconstruction_loss_type` (`"l1"` or `"mse"`)
- `encoder_loss_weight` (used when `use_encoder_loss=True`)

See `examples/research_projects/autoencoder_rae/train_autoencoder_rae.py` for a stage-1 style training script.

## AutoencoderRAE class

[[autodoc]] AutoencoderRAE
  - encode
  - decode
  - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
