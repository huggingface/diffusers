<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoRound

[AutoRound](https://github.com/intel/auto-round) is an advanced quantization toolkit. It achieves high accuracy at ultra-low bit widths (2-4 bits) with minimal tuning by leveraging sign-gradient descent and providing broad hardware compatibility. See our papers [SignRoundV1](https://arxiv.org/pdf/2309.05516) and [SignRoundV2](https://arxiv.org/abs/2512.04746) for more details.


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
from diffusers import ZImageTransformer2DModel, ZImagePipeline, AutoRoundConfig

model_id = "INCModel/Z-Image-W4A16-AutoRound"

quantization_config = AutoRoundConfig(backend="marlin")
transformer = ZImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

pipe = ZImagePipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

image = pipe("a cat holding a sign that says hello").images[0]
image.save("output.png")
```

> [!NOTE]
> AutoRound in Diffusers only supports loading *pre-quantized* models. To quantize a model from scratch, use the [AutoRound CLI or Python API](https://github.com/intel/auto-round) directly, then load the result with Diffusers.

## Backends

AutoRound supports multiple inference backends. The backend controls which kernel handles dequantization during the forward pass. Set the `backend` parameter in [`AutoRoundConfig`] to choose one:

| Backend | Value | Device | Requirements | Notes |
|---------|-------|--------|--------------|-------|
| **Auto** | `"auto"` | Any | — | Default. Automatically selects the best available backend. |
| **PyTorch** | `"torch"` | CPU / CUDA | — | Pure PyTorch implementation. Broadest compatibility. |
| **Triton** | `"tritonv2"` | CUDA | `triton` | Triton-based kernel for GPU inference. |
| **ExllamaV2** | `"exllamav2"` | CUDA | `gptqmodel>=5.8.0` | Good CUDA performance via the ExllamaV2 kernel. |
| **Marlin** | `"marlin"` | CUDA | `gptqmodel>=5.8.0` | Best CUDA performance via the Marlin kernel. |


```python
from diffusers import AutoRoundConfig

# Auto-select (default)
config = AutoRoundConfig()

# Explicit Triton backend for CUDA
config = AutoRoundConfig(backend="tritonv2")

# Marlin backend for best CUDA performance (requires gptqmodel>=5.8.0)
config = AutoRoundConfig(backend="marlin")

# Marlin backend for best CUDA performance (requires gptqmodel>=5.8.0)
config = AutoRoundConfig(backend="exllamav2")

# PyTorch backend for CPU/CUDA inference
config = AutoRoundConfig(backend="torch")
```


## Quantization configurations

AutoRound focuses on weight-only quantization. The primary configuration is W4A16 (4-bit weights, 16-bit activations), with flexibility in group size and symmetry:

| Configuration | `bits` | `group_size` | `sym` | Description |
|--------------|--------|-------------|-------|-------------|
| W4G128 asymmetric | `4` | `128` | `False` | Default. Good balance of accuracy and compression. |
| W4G128 symmetric | `4` | `128` | `True` | Faster dequantization, small accuracy trade-off. |
| W4G32 asymmetric | `4` | `32` | `False` | Higher accuracy at the cost of more metadata. |

## Save and load

<hfoptions id="save-and-load">
<hfoption id="save">

```python
from auto_round import AutoRound
autoround = AutoRound(
    tiny_z_image_model_path,
    num_inference_steps=3,
    guidance_scale=7.5,
    dataset="coco2014,
)
autoround.quantize_and_save("Z-Image-W4A16-AutoRound")
```

</hfoption>
<hfoption id="load">

```python
import torch
from diffusers import ZImageTransformer2DModel, ZImagePipeline

model_id = "INCModel/Z-Image-W4A16-AutoRound"

# The inference backend will be automatically selected.
pipe = ZImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

image = pipe("a cat holding a sign that says hello").images[0]
image.save("output.png")
```

</hfoption>
</hfoptions>

## Resources

- [Pre-quantized AutoRound models on the Hub](https://huggingface.co/models?search=autoround)
