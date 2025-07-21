<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLCogVideoX

The 3D variational autoencoder (VAE) model with KL loss used in [CogVideoX](https://github.com/THUDM/CogVideo) was introduced in [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://github.com/THUDM/CogVideo/blob/main/resources/CogVideoX.pdf) by Tsinghua University & ZhipuAI.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLCogVideoX

vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to("cuda")
```

## AutoencoderKLCogVideoX

[[autodoc]] AutoencoderKLCogVideoX
    - decode
    - encode
    - all

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
