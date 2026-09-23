<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLQwenImage21

The 64-channel variational auto-encoder used by Qwen-Image 2.1. It compresses 16x spatially, and its per-channel
`latents_mean` / `latents_std` are part of the config rather than a single scaling factor.

```python
import torch
from diffusers import AutoencoderKLQwenImage21

vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1", subfolder="vae", dtype=torch.bfloat16)
```

## AutoencoderKLQwenImage21

[[autodoc]] AutoencoderKLQwenImage21
    - decode
    - encode
    - all

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
