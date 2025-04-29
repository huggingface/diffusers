<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLMochi

The 3D variational autoencoder (VAE) model with KL loss used in [Mochi](https://github.com/genmoai/models) was introduced in [Mochi 1 Preview](https://huggingface.co/genmo/mochi-1-preview) by Tsinghua University & ZhipuAI.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLMochi

vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
```

## AutoencoderKLMochi

[[autodoc]] AutoencoderKLMochi
    - decode
    - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
