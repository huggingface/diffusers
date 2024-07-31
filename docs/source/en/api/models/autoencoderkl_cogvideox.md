<!--Copyright 2024 The The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderKLCogVideoX

The 3D variational autoencoder (VAE) model with KL loss using with CogVideoX.

The abstract from the paper is:


## Loading from the original format

By default the [`AutoencoderKL`] should be loaded with [`~ModelMixin.from_pretrained`], but it can also be loaded
from the original format using [`FromOriginalModelMixin.from_single_file`] as follows:

```py
from diffusers import AutoencoderKLCogVideoX

url = "3d-vae.pt"  # can also be a local file
model = AutoencoderKLCogVideoX.from_single_file(url)
```

## AutoencoderKLCogVideoX

[[autodoc]] AutoencoderKLCogVideoX
    - decode
    - encode
    - all
