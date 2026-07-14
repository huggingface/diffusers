<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Nunchaku Lite

Nunchaku Lite is a quantization backend for loading prequantized checkpoints in Diffusers. Create compatible checkpoints with [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor). It quantizes and exports a transformer, then packages it as a Diffusers pipeline.

Nunchaku Lite builds on the original [Nunchaku](https://github.com/nunchaku-ai/nunchaku) inference engine,
[DeepCompressor](https://github.com/nunchaku-ai/deepcompressor) quantization library, and
[SVDQuant paper](https://arxiv.org/abs/2411.05007).

## Install the CUDA kernels

The kernels package supplies the optimized CUDA kernels, which load automatically. Install it first.

```bash
pip install -U kernels
```

## Load a quantized pipeline

Load the prequantized pipeline with [`~DiffusionPipeline.from_pretrained`], which reads the quantization
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
).images[0]
image.save("ernie-image-turbo-nunchaku-lite.png")
```

> [!NOTE]
> The exported state dict must match the target Diffusers model architecture exactly. For example, a checkpoint
> quantized with fused QKV projections won't load into a model config that expects separate Q, K, and V projection
> modules.

## Supported quantization types

Nunchaku Lite supports the following quantized linear layer formats.

> [!TIP]
> Use `nvfp4` on Blackwell GPUs. Running `int4` checkpoints on Blackwell can be slower than `nvfp4`.

The CUDA kernels currently support the following NVIDIA GPU architectures:

- `sm_75` (Turing, for example RTX 2080)
- `sm_80` (Ampere, for example A100)
- `sm_86` (Ampere, for example RTX 3090 and RTX A6000)
- `sm_89` (Ada, for example RTX 4090)
- `sm_120` (Blackwell, for example RTX 5090)

> [!NOTE]
> Hopper GPUs, such as `sm_90` H100 and H200, are not currently supported.

`nvfp4` checkpoints require a Blackwell or newer NVIDIA GPU. On Blackwell GPUs, use PyTorch >= 2.7 with CUDA >= 12.8.
`int4` checkpoints require a Turing or newer NVIDIA GPU.

| Method | Precision | Group size | Notes |
|---|---:|---:|---|
| `svdq_w4a4` | `nvfp4` | 16 | Uses NVFP4 runtime kernels with SVDQ low-rank correction. |
| `svdq_w4a4` | `int4` | 64 | Uses INT4 W4A4 kernels with SVDQ low-rank correction. |
| `awq_w4a16` | `int4` | 64 | Uses INT4 weight-only AWQ-style kernels. |

## NunchakuLiteQuantizationConfig

The `config.json` file must include a [`NunchakuLiteQuantizationConfig`]. It defines the runtime
`compute_dtype` and the target modules for each Nunchaku Lite quantization method.

- `compute_dtype`: runtime dtype for floating-point buffers in quantized modules, typically `torch.bfloat16`.
- `svdq_w4a4`: SVDQ W4A4 target config with `precision`, `group_size`, `rank`, and `targets`.
- `awq_w4a16`: AWQ W4A16 target config with `precision`, `group_size`, and `targets`.

Each entry in `targets` must point to a linear layer. Diffusers swaps each `svdq_w4a4` target for an SVDQ W4A4 layer and each `awq_w4a16` target for an AWQ W4A16 layer. The example below shows the
expected shape with shortened target lists.

List each module you want to quantize under `svdq_w4a4` or `awq_w4a16`. A module can only use one method, so don't list the same target under both.

```json
{
  "_class_name": "ErnieImageTransformer2DModel",
  "quantization_config": {
    "quant_method": "nunchaku_lite",
    "compute_dtype": "bfloat16",
    "svdq_w4a4": {
      "precision": "nvfp4",
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
pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=True)
```

The compiled Nunchaku Lite NVFP4 pipeline runs 1.8x faster than the original BF16 pipeline (2.271s → 1.675s on an RTX PRO 6000).

## Resources

- [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)
- [Nunchaku installation requirements](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
