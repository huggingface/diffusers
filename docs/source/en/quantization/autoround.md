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

Load a pre-quantized AutoRound model by passing [`AutoRoundConfig`] to [`~ModelMixin.from_pretrained`]. The method works with any model that loads via [Accelerate](https://hf.co/docs/accelerate/index) and has `torch.nn.Linear` layers.

You can use [`PipelineQuantizationConfig`] to quantize specific components of a pipeline:

```python
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, AutoRoundConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": AutoRoundConfig(backend="auto")}
)
pipe = DiffusionPipeline.from_pretrained(
    "INCModel/Z-Image-W4A16-AutoRound",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

image = pipe("a cat holding a sign that says hello").images[0]
image.save("output.png")
```

Or load a quantized model component directly:

```python
import torch
from diffusers import ZImageTransformer2DModel, ZImagePipeline, AutoRoundConfig

model_id = "INCModel/Z-Image-W4A16-AutoRound"

quantization_config = AutoRoundConfig(backend="auto")
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

## torch.compile

AutoRound is compatible with [`torch.compile`](../optimization/fp16#torchcompile) for faster inference. You can compile the quantized transformer (DiT) for better performance:

```python
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, AutoRoundConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": AutoRoundConfig(backend="auto")}
)
pipe = DiffusionPipeline.from_pretrained(
    "INCModel/Z-Image-W4A16-AutoRound",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False)
```

## Backends

AutoRound supports multiple inference backends for Weight-only quantized model. The backend controls which kernel handles dequantization during the forward pass. Set the `backend` parameter in [`AutoRoundConfig`] to choose one:

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

# ExllamaV2 backend for good CUDA performance (requires gptqmodel>=5.8.0)
config = AutoRoundConfig(backend="exllamav2")

# PyTorch backend for CPU/CUDA inference
config = AutoRoundConfig(backend="torch")
```


## Save and load

<hfoptions id="save-and-load">
<hfoption id="save">

AutoRound requires data calibration to quantize a model. This is done outside of Diffusers using the [AutoRound library](https://github.com/intel/auto-round) directly:

```python
from auto_round import AutoRound

autoround = AutoRound(
    "Tongyi-MAI/Z-Image",
    scheme="W4A16",  # W4G128 symmetric
    enable_torch_compile=True,
    num_inference_steps=3,
    guidance_scale=7.5,
    dataset="coco2014",
)
autoround.quantize_and_save("Z-Image-W4A16-AutoRound")
```

For more details on calibration options, see the [AutoRound documentation](https://github.com/intel/auto-round).

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


### Supported Quantization Schemes

AutoRound supports several Schemes:

- **W4A16**(bits:4,group_size:128,sym:True,act_bits:16)
- **W8A16**(bits:8,group_size:128,sym:True,act_bits:16)
- **W3A16**(bits:3,group_size:128,sym:True,act_bits:16)
- **W2A16**(bits:2,group_size:128,sym:True,act_bits:16)
- **GGUF:Q4_K_M**(all Q*_K,Q*_0,Q*_1 provided by llamacpp are supported)
- **NVFP4**(Experimental feature, recommend exporting to `llm_compressor` format.data_type nvfp4,act_data_type nvfp4,static_global_scale,group_size 16)
- **MXFP4**(**Research feature, no real kernel**, Standard MXFP4, data_type mxfp,act_data_type mxfp,bits 4, act_bits 4, group_size 32)
- **MXINT4**(**Research feature, no real kernel**, Standard MXINT4, data_type mxint,act_data_type mxint,bits 4, act_bits 4, group_size 32)
- **MXFP4_RCEIL**(**Research feature,no real kernel**, NVIDIA's variant, data_type mxfp,act_data_type mxfp_rceil,bits 4, act_bits 4, group_size 32)
- **MXFP8**(**Research feature, no real kernel**, data_type mxfp,act_data_type mxfp_rceil,group_size 32)
- **FPW8A16**(**Research feature, no real kernel**, data_type fp8,group_size 0->per tensor )
- **FP8_STATIC**(**Research feature, no real kernel**, data_type:fp8,act_data_type:fp8,group_size -1 ->per channel, act_group_size=0->per tensor)

Besides, you could modify the `group_size`, `bits`, `sym` and many other configs you want, though there are maybe no real kernels.

## Resources

- [Pre-quantized AutoRound models on the Hub](https://huggingface.co/models?search=autoround)
