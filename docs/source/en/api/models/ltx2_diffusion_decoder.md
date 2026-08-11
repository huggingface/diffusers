<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# LTX2VideoDiffusionDecoderModel

The diffusion video decoder introduced in LTX-2.5 by Lightricks. Neighborhood-attention stages
upsample the latent into a context volume, and a final stage denoises pixels conditioned on that context.

It is a decoder, not an autoencoder: encoding stays with [`AutoencoderKLLTX2Video`], whose latent space this
consumes unchanged, so latents are interchangeable between the convolutional decoder and this one. Because it is
itself a diffusion model it is driven by [`LTX2VideoDiffusionDecodePipeline`] rather than being passed as a
pipeline's `vae`: run any LTX-2 pipeline with `output_type="latent"`, then decode.

```python
import torch
from diffusers import LTX2Pipeline, LTX2VideoDiffusionDecoderModel
from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import LTX2VideoDiffusionDecodePipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2.5", dtype=torch.bfloat16).to("cuda")
latents = pipe(prompt="a potter shaping a clay vase", output_type="latent").frames

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    "Lightricks/LTX-2.5", subfolder="diffusion_decoder", dtype=torch.bfloat16
).to("cuda")
decode_pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=pipe.scheduler)

# `denormalize=False`: `output_type="latent"` already applied the latent statistics, so applying them
# again here would scale every channel by its std a second time.
# The decoder also draws the noise it denoises, so decoding is only reproducible with a generator.
video = decode_pipe(
    latents, generator=torch.Generator("cuda").manual_seed(0), denormalize=False
).frames[0]
```

`vae` is an optional component on the decode pipeline: it is only consulted for the latent statistics when
`denormalize=True`, and the decoder carries its own, so a decode-only workflow does not have to load a second
autoencoder.

## Attention backends

The neighborhood-attention window is expressed as a `BlockMask`, so the decoder runs on the `flex` attention
backend by default and needs no extra dependency. PyTorch does not compile `flex_attention` unless you ask it to,
and uncompiled it materializes the full score matrix — which is impractical at full-resolution sequence lengths.
For those, either compile the decoder or switch to [NATTEN](https://github.com/SHI-Labs/NATTEN)'s kernels, which
are also what the original implementation uses. The processor fetches NATTEN from the Hub
([`shi-labs/natten`](https://huggingface.co/shi-labs/natten)) through the
[`kernels`](https://github.com/huggingface/kernels) package, so it needs `pip install kernels` rather than a local
NATTEN build:

```python
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor

decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
```

Fetching the kernel downloads code from the Hub, so the processor raises when remote code is disabled globally with
`DIFFUSERS_DISABLE_REMOTE_CODE=true`.

Every attention module in the decoder is the same neighborhood attention (per-stage differences like the kernel
size live on the module, not the processor), so `set_attn_processor` swaps them all with one shared instance.

Switching the *backend* (`decoder.set_attention_backend(...)`) to anything but `flex` raises: no other backend
accepts the `BlockMask`. Use the NATTEN processor above instead.

## Tiling

Tiled decoding is not supported — `enable_tiling` raises. Neighborhood attention rejects any tile smaller than its
kernel, including a short remnant tile, so tile sizes cannot be chosen freely. Batch slicing (`enable_slicing`)
works as usual.

## LTX2VideoDiffusionDecoderModel

[[autodoc]] LTX2VideoDiffusionDecoderModel
    - decode
    - encode
    - all
