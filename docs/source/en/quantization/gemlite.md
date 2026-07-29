<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# GemLite

[GemLite](https://github.com/dropbox/gemlite) is a quantization backend for loading prequantized checkpoints in
Diffusers. It replaces supported `torch.nn.Linear` layers with GemLite layers, which run packed low-bit weights
directly with GemLite kernels.

## Install GemLite

GemLite requires version 0.6.0 or later.

```bash
pip install -U "gemlite>=0.6.0"
```

## Load a quantized pipeline

GemLite only supports loading prequantized checkpoints. The quantization configuration is stored in the checkpoint's
`config.json` and read automatically by [`~DiffusionPipeline.from_pretrained`].

```python
import torch
from diffusers import DiffusionPipeline

model_id = "gabe-engineers/bonsai-image-ternary-4B-gemlite-2bit-unpacked-encoder"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="cuda",
)

image = pipe(
    prompt="A bonsai tree in a quiet ceramic studio, soft morning light",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]
image.save("bonsai-gemlite.png")
```

> [!TIP]
> `dtype` must match the `compute_dtype` in the checkpoint's GemLite quantization configuration. This
> checkpoint uses `torch.float16`.
