<!-- Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Flux3Transformer2DModel

A Transformer model for image-like data from [FLUX.3](https://bfl.ai/), for text-to-image and image-to-image generation. Three token modes share one joint sequence: the noisy image tokens, the VAE-encoded reference image tokens and the text context. Each mode first passes through its own per-mode transformer blocks, then the modes are concatenated for the joint single blocks, and the image tokens pass through the output layers. Timesteps are per token: the image tokens carry the flow timestep, the reference tokens and the text context sit at `0`.

```python
import torch
from diffusers import Flux3Transformer2DModel

transformer = Flux3Transformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.3-image-small", subfolder="transformer", dtype=torch.bfloat16
)
```

## Flux3Transformer2DModel

[[autodoc]] Flux3Transformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
