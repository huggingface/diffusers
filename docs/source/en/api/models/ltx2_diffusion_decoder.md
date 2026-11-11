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
from diffusers import LTX2Pipeline, LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", dtype=torch.bfloat16).to("cuda")  # or "mps", "xpu", "cpu"
latents = pipe(prompt="a potter shaping a clay vase", output_type="latent").frames

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    "Lightricks/LTX-2.5-Diffusers", subfolder="diffusion_decoder", dtype=torch.bfloat16
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

`decoder.enable_tiling()` decodes in overlapping tiles that are blended back together, bounding peak memory by the
tile size instead of the video size. The cheap early upsampling stages still see the full latent — only the last
upsampling stage and the diffusion stage, which dominate decode memory, run per tile — so tiling changes the output
only near tile borders. Because the diffusion stage denoises each tile separately, a tiled decode does not
reproduce the untiled result exactly; the default tile and overlap sizes match the reference implementation's.
Neighborhood attention rejects any grid smaller than its kernel, so a trailing remnant tile is merged into its
neighbor rather than decoded on its own.

## Keyframe-aware decoding

A decode can be anchored on *keyframe planes*: single-frame latents at known pixel frames, for example the generated
slots of [`LTX2DFRPipeline`]. Each plane must be the latent of a standalone one-frame clip (the VAE is causal, so a
frame encoded inside a clip is a different latent), denormalized like the video latents. Positions are pixel-frame
indices, not latent-frame indices.

Omit both arguments for a plain decode; that path does not depend on whether the checkpoint carries a keyframe tag.

```python
video = decoder.decode(
    latents,  # (B, C, F, H, W), already denormalized
    generator=torch.Generator("cuda").manual_seed(0),
    keyframe_latents=keyframe_latents,  # (B, C, P, H, W)
    keyframe_frame_indices=[24, 48],  # pixel frame of each plane
).sample
```

The planes go through the same weights as the video. In every attention layer each video position also attends to the
same spatial window on its two nearest planes, and each plane to the same window on its two nearest frames. This joint
attention runs on a built-in PyTorch implementation whatever attention processor is set, since neither FlexAttention's
block mask nor NATTEN expresses it.

Checkpoints trained for keyframe decoding also carry a learned tag added to the plane latents, which
`decoder_keyframe_type_embedding=True` creates as `decoder.type_emb`. Decoding with keyframe planes on a checkpoint that
lacks that tag raises rather than decoding the planes untagged. If a checkpoint's weights carry `decoder.type_emb` but
its config does not set the flag, pass `decoder_keyframe_type_embedding=True` to
[`LTX2VideoDiffusionDecoderModel.from_pretrained`]; otherwise the tag is dropped as an unexpected key. Converting the
original VAE with `convert_ltx2_diffusion_video_vae` sets the flag when the tag is present.

With tiling enabled each temporal tile keeps the planes inside it plus the nearest plane on each side.

### Standard pipeline

[`LTX2VideoDiffusionDecodePipeline`] forwards the same arguments. A DFR pass with `output_type="latent"` already
denormalized both the video and the slots:

```python
from diffusers import LTX2DFRPipeline, LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel

dfr = LTX2DFRPipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", dtype=torch.bfloat16)
out = dfr(prompt="...", output_type="latent", generate_slots=True)

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    "Lightricks/LTX-2.5-Diffusers",
    subfolder="diffusion_decoder",
    dtype=torch.bfloat16,
    decoder_keyframe_type_embedding=True,  # keep the trained keyframe tag `decoder.type_emb`
)
decode_pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=dfr.scheduler)
video = decode_pipe(
    out.frames,
    generator=torch.Generator("cuda").manual_seed(0),
    denormalize=False,
    keyframe_latents=out.keyframes,
    keyframe_frame_indices=out.keyframe_positions,
).frames
```

### Modular workflow

[`LTX2DiffusionVaeDecoderStep`] (the LTX-2.5 decode block) declares the same inputs. Packed denoised latents are
unpacked and denormalized as usual; `keyframe_latents` are already denormalized 5-D planes, matching
`LTX2DFRPipelineOutput.keyframes`. They are optional, so they do not change automatic workflow selection.

```python
from diffusers.modular_pipelines.ltx2.decoders import LTX2DiffusionVaeDecoderStep

decode = LTX2DiffusionVaeDecoderStep().init_pipeline("Lightricks/LTX-2.5-Diffusers")
# Keep the trained keyframe tag `decoder.type_emb`; the dict form passes the flag to the decoder only.
decode.load_components(dtype=torch.bfloat16, decoder_keyframe_type_embedding={"diffusion_decoder": True})
videos = decode(
    latents=packed_latents,
    height=512,
    width=704,
    num_frames=121,
    dtype=torch.bfloat16,
    generator=torch.Generator("cuda").manual_seed(0),
    keyframe_latents=out.keyframes,
    keyframe_frame_indices=out.keyframe_positions,
    output="videos",
)
```

## LTX2VideoDiffusionDecoderModel

[[autodoc]] LTX2VideoDiffusionDecoderModel
    - decode
    - enable_tiling
    - disable_tiling
    - all
