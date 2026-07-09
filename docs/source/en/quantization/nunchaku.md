<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Nunchaku Lite

Nunchaku Lite is a quantization backend for loading prequantized checkpoints in Diffusers. Use
[`diffuse-compressor`](https://github.com/rootonchair/diffuse-compressor) to create compatible checkpoints by choosing
or adapting a target config for the model architecture, quantizing and exporting the transformer, and packaging the
result as a Diffusers pipeline repository.

## Load a quantized pipeline

Load the packaged prequantized pipeline with [`~DiffusionPipeline.from_pretrained`]. Diffusers reads the quantization
config from `config.json`.

```python
import torch
from diffusers import DiffusionPipeline

model_id = "rootonchair/ERNIE-Image-Turbo-nunchaku-lite-nvfp4"

pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "A modern red armchair in a quiet studio, soft window light, realistic product photography"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=1.0,
    use_pe=False,
).images[0]
image.save("ernie-image-turbo-nunchaku-lite.png")
```

## Using Optimized CUDA Kernels with Nunchaku Lite

Nunchaku Lite uses optimized CUDA kernels through the `kernels` package. The kernels are loaded automatically when you
load a Nunchaku Lite checkpoint. Install the runtime dependency before loading a Nunchaku Lite checkpoint.

```bash
pip install -U kernels
```

> [!NOTE]
> The exported state dict must match the target Diffusers model architecture exactly. For example, a checkpoint
> quantized with fused QKV projections won't load into a model config that expects separate Q, K, and V projection
> modules.

## Supported Quantization Types

Nunchaku Lite supports the following quantized linear layer formats.

> [!TIP]
> Use `fp4` on Blackwell GPUs. Running `int4` checkpoints on Blackwell can be slower than `fp4`.

| Method | Precision | Group size | Notes |
|---|---:|---:|---|
| `svdq_w4a4` | `fp4` | 16 | Uses NVFP4 runtime kernels with SVDQ low-rank correction. |
| `svdq_w4a4` | `int4` | 64 | Uses INT4 W4A4 kernels with SVDQ low-rank correction. |
| `awq_w4a16` | `int4` | 64 | Uses INT4 weight-only AWQ-style kernels. |

## NunchakuLiteQuantizationConfig

The `model.json` file must include a compact [`NunchakuLiteQuantizationConfig`]. It defines the runtime
`compute_dtype` and the target modules for each Nunchaku Lite quantization method.

- `compute_dtype`: runtime dtype for floating-point buffers in quantized modules, typically `torch.bfloat16`.
- `svdq_w4a4`: SVDQ W4A4 target config with `precision`, `group_size`, `rank`, and `targets`.
- `awq_w4a16`: AWQ W4A16 target config with `precision`, `group_size`, and `targets`.

The example below shows the expected shape with shortened target lists.

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

## torch.compile

Nunchaku Lite kernels and quantized linear layers are compatible with [`torch.compile`](../optimization/fp16#torchcompile).
Compile the quantized transformer after loading the pipeline for faster inference.

```python
pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False)
```

An ERNIE-Image-Turbo benchmark on an RTX PRO 6000 reported that Nunchaku Lite NVFP4 with `torch.compile` reduced the full pipeline latency from 2.271s to 1.675s. Compared to the original BF16 pipeline, the compiled
Nunchaku Lite NVFP4 pipeline reached a 1.8x speedup.

## Resources

- [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)
