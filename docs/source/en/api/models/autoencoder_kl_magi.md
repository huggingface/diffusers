<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLMagi

The 3D variational autoencoder (VAE) model with KL loss used in [MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/abs/2505.13211) by Sand.ai.

MAGI-1 uses a transformer-based VAE with 8x spatial and 4x temporal compression, providing fast average decoding time and highly competitive reconstruction quality.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLMagi

vae = AutoencoderKLMagi.from_pretrained("sand-ai/MAGI-1", subfolder="vae", torch_dtype=torch.float32)
```

## AutoencoderKLMagi

[[autodoc]] AutoencoderKLMagi
  - decode
  - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput