<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLJoyVideoEdit

The causal, chunk-streamable 3D variational autoencoder (VAE) model with KL loss used in [`JoyVideoEditPipeline`]. It
encodes and decodes video in temporal chunks so that arbitrarily long sequences can be processed with bounded memory.

The model can be loaded with the following code snippet.

```python
import torch

from diffusers import AutoencoderKLJoyVideoEdit

vae = AutoencoderKLJoyVideoEdit.from_pretrained(
    "jdopensource/JoyAI-Video-Edit-Diffusers", subfolder="vae", dtype=torch.float32
)
```

## AutoencoderKLJoyVideoEdit

[[autodoc]] AutoencoderKLJoyVideoEdit
  - decode
  - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
