<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# PAN2Transformer3DModel

The diffusion transformer of PAN 2. It runs joint attention over the video tokens and the text tokens, with separate projections and feed-forward networks for each stream.

The video input stacks the noisy latents, the conditioning latents and a one-channel conditioning mask, so the same model serves text-to-video and image-to-video.

```python
import torch
from diffusers import PAN2Transformer3DModel

transformer = PAN2Transformer3DModel.from_pretrained("IFM/PAN2", subfolder="transformer", dtype=torch.bfloat16)
```

## PAN2Transformer3DModel

[[autodoc]] PAN2Transformer3DModel
