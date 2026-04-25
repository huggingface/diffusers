<!-- Copyright 2026 The NYU Vision-X and HuggingFace Teams. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderRAE

The Representation Autoencoder (RAE) model introduced in [Diffusion Transformers with Representation Autoencoders](https://huggingface.co/papers/2510.11690) by Boyang Zheng, Nanye Ma, Shengbang Tong, Saining Xie from NYU VISIONx.

RAE combines a frozen pretrained vision encoder (DINOv2, SigLIP2, or MAE) with a trainable ViT-MAE-style decoder. In the two-stage RAE training recipe, the autoencoder is trained in stage 1 (reconstruction), and then a diffusion model is trained on the resulting latent space in stage 2 (generation).

The following RAE models are released and supported in Diffusers:

| Model | Encoder | Latent shape (224px input) |
|:------|:--------|:---------------------------|
| [`nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`](https://huggingface.co/nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08) | DINOv2-base | 768 x 16 x 16 |
| [`nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08-i512`](https://huggingface.co/nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08-i512) | DINOv2-base (512px) | 768 x 32 x 32 |
| [`nyu-visionx/RAE-dinov2-wReg-small-ViTXL-n08`](https://huggingface.co/nyu-visionx/RAE-dinov2-wReg-small-ViTXL-n08) | DINOv2-small | 384 x 16 x 16 |
| [`nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08`](https://huggingface.co/nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08) | DINOv2-large | 1024 x 16 x 16 |
| [`nyu-visionx/RAE-siglip2-base-p16-i256-ViTXL-n08`](https://huggingface.co/nyu-visionx/RAE-siglip2-base-p16-i256-ViTXL-n08) | SigLIP2-base | 768 x 16 x 16 |
| [`nyu-visionx/RAE-mae-base-p16-ViTXL-n08`](https://huggingface.co/nyu-visionx/RAE-mae-base-p16-ViTXL-n08) | MAE-base | 768 x 16 x 16 |

## Loading a pretrained model

```python
from diffusers import AutoencoderRAE

model = AutoencoderRAE.from_pretrained(
    "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08"
).to("cuda").eval()
```

## Encoding and decoding a real image

```python
import torch
from diffusers import AutoencoderRAE
from diffusers.utils import load_image
from torchvision.transforms.functional import to_tensor, to_pil_image

model = AutoencoderRAE.from_pretrained(
    "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08"
).to("cuda").eval()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
image = image.convert("RGB").resize((224, 224))
x = to_tensor(image).unsqueeze(0).to("cuda")  # (1, 3, 224, 224), values in [0, 1]

with torch.no_grad():
    latents = model.encode(x).latent        # (1, 768, 16, 16)
    recon = model.decode(latents).sample     # (1, 3, 256, 256)

recon_image = to_pil_image(recon[0].clamp(0, 1).cpu())
recon_image.save("recon.png")
```

## Latent normalization

Some pretrained checkpoints include per-channel `latents_mean` and `latents_std` statistics for normalizing the latent space. When present, `encode` and `decode` automatically apply the normalization and denormalization, respectively.

```python
model = AutoencoderRAE.from_pretrained(
    "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08"
).to("cuda").eval()

# Latent normalization is handled automatically inside encode/decode
# when the checkpoint config includes latents_mean/latents_std.
with torch.no_grad():
    latents = model.encode(x).latent   # normalized latents
    recon = model.decode(latents).sample
```

## AutoencoderRAE

[[autodoc]] AutoencoderRAE
  - encode
  - decode
  - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
