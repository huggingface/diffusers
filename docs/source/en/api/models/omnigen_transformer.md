<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# OmniGenTransformer2DModel

A Transformer model that accepts multimodal instructions to generate images for [OmniGen](https://github.com/VectorSpaceLab/OmniGen/).

The abstract from the paper is:

*The emergence of Large Language Models (LLMs) has unified language  generation tasks and revolutionized human-machine interaction.  However, in the realm of image generation, a unified model capable of handling various tasks within a single framework remains largely unexplored. In this work, we introduce OmniGen, a new diffusion model for unified image generation. OmniGen is characterized by the following features: 1) Unification: OmniGen not only demonstrates text-to-image generation capabilities but also inherently supports various downstream tasks, such as image editing, subject-driven generation, and visual conditional generation. 2) Simplicity: The architecture of OmniGen is highly simplified, eliminating the need for additional plugins. Moreover, compared to existing diffusion models, it is more user-friendly and can complete complex tasks end-to-end through instructions without the need for extra intermediate steps, greatly simplifying the image generation workflow. 3) Knowledge Transfer: Benefit from learning in a unified format, OmniGen effectively transfers knowledge across different tasks, manages unseen tasks and domains, and exhibits novel capabilities. We also explore the model’s reasoning capabilities and potential applications of the chain-of-thought mechanism.  This work represents the first attempt at a general-purpose image generation model,  and we will release our resources at https://github.com/VectorSpaceLab/OmniGen to foster future advancements.*

```python
import torch
from diffusers import OmniGenTransformer2DModel

transformer = OmniGenTransformer2DModel.from_pretrained("Shitao/OmniGen-v1-diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## OmniGenTransformer2DModel

[[autodoc]] OmniGenTransformer2DModel
