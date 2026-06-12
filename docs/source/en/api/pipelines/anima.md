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

# Anima

Anima is a text-to-image model that reuses the [`CosmosTransformer3DModel`] with a Qwen3 text encoder, a T5-token text conditioner, and the [`AutoencoderKLQwenImage`] VAE.

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers")
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(prompt="masterpiece, best quality, 1girl, solo, city lights").images[0]
```

## AnimaModularPipeline

[[autodoc]] AnimaModularPipeline

## AnimaAutoBlocks

[[autodoc]] AnimaAutoBlocks

## AnimaImg2ImgAutoBlocks

[[autodoc]] AnimaImg2ImgAutoBlocks

## AnimaTextConditioner

[[autodoc]] AnimaTextConditioner
