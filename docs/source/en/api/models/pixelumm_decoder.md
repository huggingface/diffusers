<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# PixelUMMDecoder

PixelUMM is a distilled diffusion decoder for 32-channel video latents, with spatial compression 32 and temporal
compression 4. It predicts clean pixels conditioned on the latent video and an independently sampled noise tensor.
It does not encode videos into latents.

```python
import torch
from diffusers import PixelUMMDecoder

decoder = PixelUMMDecoder.from_pretrained("tencent/Hy-Nomas-Diffusers", subfolder="decoder", torch_dtype=torch.bfloat16)
```

The decoder supports local spatial attention, overlapping temporal windows, optional spatial tiling, and explicit
streaming sessions. Noise is indexed by global latent frame, allowing overlapping windows to use consistent noise.

## PixelUMMDecoder

[[autodoc]] PixelUMMDecoder
    - forward
    - decode
    - create_streaming_decode_session
