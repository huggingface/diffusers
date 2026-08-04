<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLLTX2VideoDiffusionDecoder

The video VAE used from LTX-2.4 onwards, introduced by Lightricks. It pairs the causal convolutional encoder of
[`AutoencoderKLLTX2Video`] — same weights, same latent space — with a *diffusion* decoder: neighborhood-attention
stages upsample the latent into a context volume, and a final stage denoises pixels conditioned on that context.
Because the encoder is unchanged, latents are interchangeable between the two classes and either decoder can
consume them.

The diffusion decoder ships as a second subfolder alongside the convolutional `vae`, so it is opt-in:

```python
import torch
from diffusers import AutoencoderKLLTX2VideoDiffusionDecoder, LTX2Pipeline

vae = AutoencoderKLLTX2VideoDiffusionDecoder.from_pretrained(
    "Lightricks/LTX-2.4", subfolder="vae_diffusion", dtype=torch.bfloat16
)
pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2.4", vae=vae, dtype=torch.bfloat16)
```

Unlike a convolutional decoder this one draws noise and denoises it, so decoding is only reproducible when a
generator is passed. `LTX2Pipeline` forwards the generator it was called with; calling `decode` directly, pass
your own:

```python
video = vae.decode(latents, generator=torch.Generator("cuda").manual_seed(0)).sample
```

## Attention backends

The neighborhood-attention window is expressed as a `BlockMask`, so the decoder runs on the `flex` attention
backend by default and needs no extra dependency. PyTorch does not compile `flex_attention` unless you ask it to,
and uncompiled it materializes the full score matrix — which is impractical at full-resolution sequence lengths.
For those, either compile the decoder or install [`natten`](https://github.com/SHI-Labs/NATTEN) and switch to its
kernels, which are also what the original implementation uses:

```python
from diffusers.models.autoencoders.autoencoder_kl_ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodAttention,
    LTX2VideoVaeNeighborhoodNattenProcessor,
)

for module in vae.modules():
    if isinstance(module, LTX2VideoVaeNeighborhoodAttention):
        module.set_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
```

Switching the *backend* (`vae.set_attention_backend(...)`) to anything but `flex` raises: no other backend
accepts the `BlockMask`. Use the NATTEN processor above instead.

## Tiling

Tiled decoding is not supported — `enable_tiling` raises. Neighborhood attention rejects any tile smaller than its
kernel, including a short remnant tile, so tile sizes cannot be chosen freely. Batch slicing (`enable_slicing`)
works as usual.

## AutoencoderKLLTX2VideoDiffusionDecoder

[[autodoc]] AutoencoderKLLTX2VideoDiffusionDecoder
    - decode
    - encode
    - all
