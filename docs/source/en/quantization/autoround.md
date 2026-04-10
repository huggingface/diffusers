<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoRound

[AutoRound](https://github.com/intel/auto-round) is a weight-only quantization algorithm that uses **S**ign **G**radient **D**escent to jointly optimize rounding values and min-max ranges for weights. It targets the W4A16 configuration (4-bit weights, 16-bit activations), reducing model memory footprint while preserving inference accuracy.

> **Paper:** [AutoRound: Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://huggingface.co/papers/2309.05516)

Before you begin, make sure you have `auto-round` installed (version ≥ 0.13.0):

```bash
pip install "auto-round>=0.13.0"
```

For best CUDA inference performance with the Marlin kernel, also install `gptqmodel`:

```bash
pip install "gptqmodel>=5.8.0"
```

## Quickstart

Load a pre-quantized AutoRound model by passing [`AutoRoundConfig`] to [`~ModelMixin.from_pretrained`]. This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

```python
import torch
from diffusers import AutoModel, FluxPipeline, AutoRoundConfig

model_id = "your-org/flux-autoround-w4g128"

quantization_config = AutoRoundConfig(bits=4, group_size=128, sym=False)
transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)
pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

image = pipe("A cat holding a sign that says hello world").images[0]
image.save("output.png")
```

> **Note:** AutoRound in Diffusers only supports loading **pre-quantized** models. To quantize a model from scratch, use the [AutoRound CLI or Python API](https://github.com/intel/auto-round) directly, then load the result with Diffusers.

## Inference backends

AutoRound supports multiple inference backends. The backend determines which kernel is used for dequantization during the forward pass. Set the `backend` parameter in [`AutoRoundConfig`] to choose one:

| Backend | Value | Device | Requirements | Notes |
|---------|-------|--------|--------------|-------|
| **Auto** | `"auto"` | Any | — | Default. Automatically selects the best available backend. |
| **PyTorch** | `"auto_round:torch_zp"` | CPU / CUDA | — | Pure PyTorch implementation. Broadest compatibility. |
| **Triton** | `"auto_round:tritonv2_zp"` | CUDA | `triton` | Triton-based kernel for GPU inference. |
| **Marlin** | `"gptqmodel:marlin_zp"` | CUDA | `gptqmodel>=5.8.0` | Best CUDA performance via the Marlin kernel. |

### Example: specifying a backend

```python
from diffusers import AutoRoundConfig

# Auto-select (default)
config = AutoRoundConfig(bits=4, group_size=128)

# Explicit Triton backend for CUDA
config = AutoRoundConfig(bits=4, group_size=128, backend="auto_round:tritonv2_zp")

# Marlin backend for best CUDA performance (requires gptqmodel>=5.8.0)
config = AutoRoundConfig(bits=4, group_size=128, backend="gptqmodel:marlin_zp")

# PyTorch backend for CPU inference
config = AutoRoundConfig(bits=4, group_size=128, backend="auto_round:torch_zp")
```

## AutoRoundConfig

The [`AutoRoundConfig`] class accepts the following parameters:

- `bits` (`int`, defaults to `4`): Number of bits for weight quantization. Use `4` for W4A16.
- `group_size` (`int`, defaults to `128`): Group size for quantization. Weights in each group share the same scale and zero-point. Common values: `32`, `64`, `128`, or `-1` (per-channel).
- `sym` (`bool`, defaults to `False`): Whether to use symmetric quantization (`True`) or asymmetric quantization (`False`). Asymmetric is generally more accurate.
- `modules_to_not_convert` (`list[str]` or `None`, defaults to `None`): List of module name patterns to exclude from quantization. Use this to keep sensitive layers in full precision.
- `backend` (`str`, defaults to `"auto"`): The inference backend kernel. See the [Inference backends](#inference-backends) table above.

## Supported quantization configurations

AutoRound focuses on weight-only quantization. The primary configuration is W4A16 (4-bit weights, 16-bit activations), with flexibility in group size and symmetry:

| Configuration | `bits` | `group_size` | `sym` | Description |
|--------------|--------|-------------|-------|-------------|
| W4G128 asymmetric | `4` | `128` | `False` | Default. Good balance of accuracy and compression. |
| W4G128 symmetric | `4` | `128` | `True` | Slightly faster dequantization, marginal accuracy loss. |
| W4G32 asymmetric | `4` | `32` | `False` | Finer granularity, better accuracy, slightly more metadata overhead. |

## Serializing and deserializing quantized models

AutoRound quantized models can be saved and reloaded using the standard [`~ModelMixin.save_pretrained`] and [`~ModelMixin.from_pretrained`] methods.

### Save

```python
import torch
from diffusers import AutoModel, AutoRoundConfig

model_id = "your-org/flux-autoround-w4g128"
quantization_config = AutoRoundConfig(bits=4, group_size=128, sym=False)
model = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)
model.save_pretrained("path/to/saved_model")
```

### Load

```python
import torch
from diffusers import AutoModel, FluxPipeline

transformer = AutoModel.from_pretrained(
    "path/to/saved_model",
    torch_dtype=torch.float16,
)
pipe = FluxPipeline.from_pretrained(
    "your-org/flux-autoround-w4g128",
    transformer=transformer,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

image = pipe("A beautiful sunset over the ocean").images[0]
image.save("output.png")
```

## Resources

- [AutoRound GitHub repository](https://github.com/intel/auto-round)
- [AutoRound paper (arXiv:2309.05516)](https://arxiv.org/abs/2309.05516)
- [AutoRound Hugging Face integration (Transformers)](https://huggingface.co/docs/transformers/quantization/autoround)
- [Pre-quantized AutoRound models on the Hub](https://huggingface.co/models?search=autoround)
