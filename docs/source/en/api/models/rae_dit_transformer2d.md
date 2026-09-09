<!-- Copyright 2026 The NYU Vision-X and HuggingFace Teams. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RAEDiT2DModel

The `RAEDiT2DModel` is the Stage-2 latent diffusion transformer introduced in
[Diffusion Transformers with Representation Autoencoders](https://huggingface.co/papers/2510.11690).

Unlike DiT models that operate on VAE latents, this transformer denoises the latent space learned by
[`AutoencoderRAE`](./autoencoder_rae). It is designed to be used with [`FlowMatchEulerDiscreteScheduler`] and
decoded back to RGB with [`AutoencoderRAE`].

## Loading a pretrained transformer

```python
from diffusers import RAEDiT2DModel

transformer = RAEDiT2DModel.from_pretrained("path/to/converted-stage2-transformer")
```

## RAEDiT2DModel

[[autodoc]] RAEDiT2DModel
