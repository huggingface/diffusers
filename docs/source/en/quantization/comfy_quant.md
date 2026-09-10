<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Comfy Quant

The [Comfy Quant](https://github.com/Comfy-Org/comfy-quants) toolkit provides state-of-the-art quantization techniques. While `comfy-quants` is used for exporting and quantizing models, Diffusers natively supports running inference on these models using the [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) library. 

`comfy-kitchen` provides highly optimized GPU kernels that allow you to seamlessly run quantized layers. By passing a `ComfyQuantConfig` to Diffusers, the library will dynamically intercept parameters and wrap them in a `QuantizedTensor` that maps directly to the optimized `comfy-kitchen` layouts.

Before starting, please install `comfy-kitchen` in your environment:

```shell
pip install comfy-kitchen
```

## Loading a Comfy Quant Model

To load a model prequantized with Comfy Quant, use the [`~FromSingleFileMixin.from_single_file`] method and pass in the [`ComfyQuantConfig`]. 

The configuration requires you to specify the `quant_format` that the model was quantized in, and the `compute_dtype` for active inference calculations.

The following example demonstrates how to load a quantized FLUX transformer:

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, ComfyQuantConfig

ckpt_path = "path/to/comfy_quant_checkpoint.safetensors"

# Initialize the config with your desired format and compute dtype
quantization_config = ComfyQuantConfig(
    quant_format="fp8",
    compute_dtype=torch.bfloat16
)

# Load the transformer directly from the safetensors file
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
)

# Pass the quantized transformer into the pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
image.save("flux-comfy-quant.png")
```

## Supported Quantization Formats

Diffusers currently maps the following Comfy Quant formats to `comfy-kitchen` layouts:

- **FP8** (`fp8`): Maps to `TensorCoreFP8Layout` (E4M3/E5M2)
- **INT8** (`int8`): Maps to `TensorCoreInt8Layout` (W8A8, tensorwise)
- **MXFP8** (`mxfp8`): Maps to `TensorCoreMXFP8Layout`
- **NVFP4** (`nvfp4`): Maps to `TensorCoreNVFP4Layout`
- **INT4 SVD** (`int4_svd`): Maps to `SVDQuantW4A4Layout` (SVDQuant W4A4)
- **INT4 AWQ** (`int4_awq`): Maps to `AWQW4A16Layout` (AWQ W4A16)

When using optimized layouts, `comfy-kitchen` automatically dispatches the operations to the best available backend (HIP, CUDA, Triton, or Eager).
