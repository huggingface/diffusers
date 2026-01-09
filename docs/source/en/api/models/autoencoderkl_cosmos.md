<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLCosmos

[Cosmos Tokenizers](https://github.com/NVIDIA/Cosmos-Tokenizer).

Supported models:
- [nvidia/Cosmos-1.0-Tokenizer-CV8x8x8](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-CV8x8x8)

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLCosmos

vae = AutoencoderKLCosmos.from_pretrained("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8", subfolder="vae")
```

## AutoencoderKLCosmos

[[autodoc]] AutoencoderKLCosmos
    - decode
    - encode
    - all

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
