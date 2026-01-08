<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLLTX2Audio

The 3D variational autoencoder (VAE) model with KL loss used in [LTX-2](https://huggingface.co/Lightricks/LTX-2) was introduced by Lightricks. This is for encoding and decoding audio latent representations.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLLTX2Audio

vae = AutoencoderKLLTX2Audio.from_pretrained("Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.float32).to("cuda")
```

## AutoencoderKLLTX2Audio

[[autodoc]] AutoencoderKLLTX2Audio
    - encode
    - decode
    - all