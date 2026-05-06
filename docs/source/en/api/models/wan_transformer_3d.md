<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# WanTransformer3DModel

A Diffusion Transformer model for 3D video-like data was introduced in [Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

The model can be loaded with the following code snippet.

```python
from diffusers import WanTransformer3DModel

transformer = WanTransformer3DModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## WanTransformer3DModel

[[autodoc]] WanTransformer3DModel

## Rolling KV cache

For autoregressive video generation that produces one chunk at a time, [`WanTransformer3DModel.forward`] accepts a `WanKVCache` instance via `attention_kwargs={"kv_cache": cache}`. The cache holds post-norm, post-RoPE self-attention K/V tensors from prior chunks so subsequent chunks attend over the full prefix without recomputing it. The chunk's RoPE positions are picked via the `frame_offset` argument on `forward`.

The cache exposes two write modes that the caller toggles between denoising steps:

- `enable_append_mode()` — the next forward pass appends the chunk's K/V to the cache; once the cache reaches `window_size`, the oldest tokens are evicted from the front. Use this for the first denoising step of every new chunk.
- `enable_overwrite_mode()` — the next forward pass replaces the newest `chunk_size` tokens in place. Use this for subsequent denoising steps within the same chunk so re-running the chunk doesn't grow the cache.

```python
from diffusers import WanKVCache, WanTransformer3DModel

transformer = WanTransformer3DModel.from_pretrained(...)
cache = WanKVCache(num_blocks=len(transformer.blocks))

for chunk_idx, latent_chunk in enumerate(chunks):
    for step_idx, t in enumerate(denoising_steps):
        if step_idx == 0:
            cache.enable_append_mode()
        else:
            cache.enable_overwrite_mode()
        transformer(
            hidden_states=latent_chunk,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            frame_offset=chunk_idx * patch_frames_per_chunk,
            attention_kwargs={"kv_cache": cache},
        )

cache.reset()  # between videos
```

## WanKVCache

[[autodoc]] WanKVCache

## WanKVBlockCache

[[autodoc]] WanKVBlockCache

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
