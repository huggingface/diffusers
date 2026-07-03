<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Nunchaku Lite

Nunchaku Lite support in Diffusers loads prequantized checkpoints. Use
[`diffuse-compressor`](https://github.com/rootonchair/diffuse-compressor) to create compatible checkpoints: choose or
adapt a target config for the model architecture, quantize and export the transformer, and package the result as a
Diffusers pipeline repository.

The exported state dict must match the target Diffusers model architecture exactly. For example, a checkpoint
quantized with fused QKV projections won't load into a model config that expects separate Q, K, and V projection
modules.

Install the runtime dependency before loading a Nunchaku Lite checkpoint.

```bash
pip install -U kernels
```

The `model.json` file must include a compact [`NunchakuLiteQuantizationConfig`]. The example below shows the expected
shape with shortened target lists.

```json
{
  "_class_name": "ErnieImageTransformer2DModel",
  "quantization_config": {
    "quant_method": "nunchaku_lite",
    "compute_dtype": "bfloat16",
    "svdq_w4a4": {
      "precision": "fp4",
      "group_size": 16,
      "rank": 32,
      "targets": ["layers.0.self_attention.to_q"]
    },
    "awq_w4a16": {
      "precision": "int4",
      "group_size": 64,
      "targets": ["final_linear"]
    }
  }
}
```

Load the packaged prequantized pipeline with [`~DiffusionPipeline.from_pretrained`]. Diffusers reads the quantization
config from `model.json`.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "path-or-hub-id-to-prequantized-pipeline",
    torch_dtype=torch.bfloat16,
).to("cuda")
```
