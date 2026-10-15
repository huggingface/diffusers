<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 6 Transformers

Kandinsky 6 uses a multimodal diffusion transformer that denoises video and audio latents together for
text/image-to-video-and-audio generation, and a text-free diffusion transformer for video super-resolution.

## Kandinsky6Transformer3DModel

The multimodal transformer used by [`Kandinsky6TI2VAPipeline`].

```python
import torch
from diffusers import Kandinsky6Transformer3DModel

transformer = Kandinsky6Transformer3DModel.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
```

[[autodoc]] Kandinsky6Transformer3DModel
  - all
  - forward

## Kandinsky6SRTransformer3DModel

The text-free transformer used by [`Kandinsky6SRPipeline`] to refine one tile of the upscaled video at a time.

```python
import torch
from diffusers import Kandinsky6SRTransformer3DModel

transformer = Kandinsky6SRTransformer3DModel.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
# The transformer always runs NABLA sparse attention (`nabla_threshold`, 0.8 by default) on the `flex` backend.
# Compile it, otherwise flex falls back to an eager implementation that needs far more memory at video resolutions.
transformer.compile_repeated_blocks(fullgraph=True)
```

[[autodoc]] Kandinsky6SRTransformer3DModel
  - all
  - forward
