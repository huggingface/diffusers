<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Tiny Video AutoEncoder

Tiny AutoEncoder for Hunyuan Video (TAEHV) was introduced in [madebyollin/taehv](https://github.com/madebyollin/taehv) by Ollin Boer Bohan. It is a family of tiny causal video autoencoders distilled from full video VAEs — `taew2_2` decodes the Wan 2.2 latent space of [`AutoencoderKLWan`] about 50× faster than the full model — for previews and real-time decoding. Latents are the normalized (roughly unit Gaussian) latents of the full VAE.

Decode a video chunk by chunk with a [`TinyVideoDecodeCache`]: each call decodes only the new latent frames, continuing from the previous calls, and the result is identical to a single decode of all frames.

```python
import torch
from diffusers import AutoencoderTinyVideo
from diffusers.models.autoencoders.autoencoder_tiny_video import TinyVideoDecodeCache

vae = AutoencoderTinyVideo.from_pretrained("YiYiXu/taew2_2-diffusers", dtype=torch.bfloat16).to("cuda")

cache = TinyVideoDecodeCache()
for latents in latent_chunks:  # [B, 48, T, h, w], normalized Wan 2.2 latents
    frames = vae.decode(latents, cache=cache).sample  # [B, 3, 4 * T, 16 * h, 16 * w] in [-1, 1]
```

## AutoencoderTinyVideo

[[autodoc]] AutoencoderTinyVideo

## TinyVideoDecodeCache

[[autodoc]] models.autoencoders.autoencoder_tiny_video.TinyVideoDecodeCache
