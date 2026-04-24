<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoRound

[AutoRound](https://huggingface.co/papers/2309.05516) is a weight-only quantization algorithm. It uses sign gradient descent to jointly optimize weight rounding and min-max ranges. AutoRound targets W4A16 (4-bit weights, 16-bit activations), reducing memory usage without sacrificing inference accuracy.


Install `auto-round`(version ≥ 0.13.0):

```bash
pip install "auto-round>=0.13.0"
```

To use the Marlin kernel for faster CUDA inference, install `gptqmodel`:

```bash
pip install "gptqmodel>=5.8.0"
```

## Load a quantized model

Load a pre-quantized AutoRound model by passing [`AutoRoundConfig`] to [`~ModelMixin.from_pretrained`]. The method works with any model that loads via [Accelerate(https://hf.co/docs/accelerate/index) and has `torch.nn.Linear` layers.

```python
import torch
from diffusers import AutoModel, FluxPipeline, AutoRoundConfig

model_id = "INCModel/Z-Image-W4A16-AutoRound"

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

> [!NOTE]
> AutoRound in Diffusers only supports loading *pre-quantized* models. To quantize a model from scratch, use the [AutoRound CLI or Python API](https://github.com/intel/auto-round) directly, then load the result with Diffusers.

## Backends

AutoRound supports multiple inference backends. The backend controls which kernel handles dequantization during the forward pass. Set the `backend` parameter in [`AutoRoundConfig`] to choose one:

| Backend | Value | Device | Requirements | Notes |
|---------|-------|--------|--------------|-------|
| **Auto** | `"auto"` | Any | — | Default. Automatically selects the best available backend. |
| **PyTorch** | `"auto_round:torch_zp"` | CPU / CUDA | — | Pure PyTorch implementation. Broadest compatibility. |
| **Triton** | `"auto_round:tritonv2_zp"` | CUDA | `triton` | Triton-based kernel for GPU inference. |
| **Marlin** | `"gptqmodel:marlin_zp"` | CUDA | `gptqmodel>=5.8.0` | Best CUDA performance via the Marlin kernel. |


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

## Quantization configurations

AutoRound focuses on weight-only quantization. The primary configuration is W4A16 (4-bit weights, 16-bit activations), with flexibility in group size and symmetry:

| Configuration | `bits` | `group_size` | `sym` | Description |
|--------------|--------|-------------|-------|-------------|
| W4G128 asymmetric | `4` | `128` | `False` | Default. Good balance of accuracy and compression. |
| W4G128 symmetric | `4` | `128` | `True` | Faster dequantization, small accuracy trade-off. |
| W4G32 asymmetric | `4` | `32` | `False` | Higher accuracy at the cost of more metadata. |

## Save and load

Save and reload AutoRound quantized models using the standard [`~ModelMixin.save_pretrained`] and [`~ModelMixin.from_pretrained`] methods.

### Save

```python
import torch
from diffusers import AutoModel, AutoRoundConfig

model_id = "INCModel/Z-Image-W4A16-AutoRound"
quantization_config = AutoRoundConfig(bits=4, group_size=128, sym=False)
model = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)
model.save_pretrained("path/to/saved_model")
```


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

- [AutoRound Hugging Face integration (Transformers)](https://huggingface.co/docs/transformers/quantization/autoround)
- [Pre-quantized AutoRound models on the Hub](https://huggingface.co/models?search=autoround)
