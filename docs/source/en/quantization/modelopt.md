<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# NVIDIA ModelOpt

[NVIDIA-ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) is a unified library of state-of-the-art model optimization techniques like quantization, pruning, distillation, speculative decoding, etc. It compresses deep learning models for downstream deployment frameworks like TensorRT-LLM or TensorRT to optimize inference speed.

Before you begin, make sure you have nvidia_modelopt installed.

```bash
pip install -U "nvidia_modelopt[hf]"
```

Quantize a model by passing [`NVIDIAModelOptConfig`] to [`~ModelMixin.from_pretrained`] (you can also load pre-quantized models). This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

The example below only quantizes the weights to FP8.

```python
import torch
from diffusers import AutoModel, SanaPipeline, NVIDIAModelOptConfig

model_id = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
dtype = torch.bfloat16

quantization_config = NVIDIAModelOptConfig(quant_type="FP8", quant_method="modelopt")
transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
pipe = SanaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=dtype,
)
pipe.to("cuda")

print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt, num_inference_steps=50, guidance_scale=4.5, max_sequence_length=512
).images[0]
image.save("output.png")
```

> **Note:**
>
> The quantization methods in NVIDIA-ModelOpt are designed to reduce the memory footprint of model weights using various QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization) techniques while maintaining model performance. However, the actual performance gain during inference depends on the deployment framework (e.g., TRT-LLM, TensorRT) and the specific hardware configuration.  
> 
> More details can be found [here](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples).

## NVIDIAModelOptConfig

The `NVIDIAModelOptConfig` class accepts three parameters:
- `quant_type`: A string value mentioning one of the quantization types below.
- `modules_to_not_convert`: A list of module full/partial module names for which quantization should not be performed. For example, to not perform any quantization of the [`SD3Transformer2DModel`]'s pos_embed projection blocks, one would specify: `modules_to_not_convert=["pos_embed.proj.weight"]`.
- `disable_conv_quantization`: A boolean value which when set to `True` disables quantization for all convolutional layers in the model. This is useful as channel and block quantization generally don't work well with convolutional layers (used with INT4, NF4, NVFP4). If you want to disable quantization for specific convolutional layers, use `modules_to_not_convert` instead.
- `algorithm`: The algorithm to use for determining scale, defaults to `"max"`. You can check modelopt documentation for more algorithms and details.
- `forward_loop`: The forward loop function to use for calibrating activation during quantization. If not provided, it relies on static scale values computed using the weights only.
- `kwargs`: A dict of keyword arguments to pass to the underlying quantization method which will be invoked based on `quant_type`.

## Supported quantization types

ModelOpt supports weight-only, channel and block quantization int8, fp8, int4, nf4, and nvfp4. The quantization methods are designed to reduce the memory footprint of the model weights while maintaining the performance of the model during inference.

Weight-only quantization stores the model weights in a specific low-bit data type but performs computation with a higher-precision data type, like `bfloat16`. This lowers the memory requirements from model weights but retains the memory peaks for activation computation.

The quantization methods supported are as follows:

| **Quantization Type** | **Supported Schemes** | **Required Kwargs** | **Additional Notes** |
|-----------------------|-----------------------|---------------------|----------------------|
| **INT8** | `int8 weight only`, `int8 channel quantization`, `int8 block quantization` | `quant_type`, `quant_type + channel_quantize`, `quant_type + channel_quantize + block_quantize` |
| **FP8** | `fp8 weight only`, `fp8 channel quantization`, `fp8 block quantization` | `quant_type`, `quant_type + channel_quantize`, `quant_type + channel_quantize + block_quantize` |
| **INT4** | `int4 weight only`, `int4 block quantization` | `quant_type`, `quant_type + channel_quantize + block_quantize` | `channel_quantize = -1 is only supported for now`|
| **NF4** | `nf4 weight only`, `nf4 double block quantization` | `quant_type`, `quant_type + channel_quantize + block_quantize + scale_channel_quantize` + `scale_block_quantize` | `channel_quantize = -1 and scale_channel_quantize = -1 are only supported for now` |
| **NVFP4** | `nvfp4 weight only`, `nvfp4 block quantization` | `quant_type`, `quant_type + channel_quantize + block_quantize` | `channel_quantize = -1 is only supported for now`|


Refer to the [official modelopt documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/) for a better understanding of the available quantization methods and the exhaustive list of configuration options available.

## Serializing and Deserializing quantized models

To serialize a quantized model in a given dtype, first load the model with the desired quantization dtype and then save it using the [`~ModelMixin.save_pretrained`] method.

```python
import torch
from diffusers import AutoModel, NVIDIAModelOptConfig
from modelopt.torch.opt import enable_huggingface_checkpointing

enable_huggingface_checkpointing()

model_id = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
quant_config_fp8 = {"quant_type": "FP8", "quant_method": "modelopt"}
quant_config_fp8 = NVIDIAModelOptConfig(**quant_config_fp8)
model = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config_fp8,
    torch_dtype=torch.bfloat16,
)
model.save_pretrained('path/to/sana_fp8', safe_serialization=False)
```

To load a serialized quantized model, use the [`~ModelMixin.from_pretrained`] method.

```python
import torch
from diffusers import AutoModel, NVIDIAModelOptConfig, SanaPipeline
from modelopt.torch.opt import enable_huggingface_checkpointing

enable_huggingface_checkpointing()

quantization_config = NVIDIAModelOptConfig(quant_type="FP8", quant_method="modelopt")
transformer = AutoModel.from_pretrained(
    "path/to/sana_fp8",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt, num_inference_steps=50, guidance_scale=4.5, max_sequence_length=512
).images[0]
image.save("output.png")
```
