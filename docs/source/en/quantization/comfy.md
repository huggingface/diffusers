<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Comfy Quantization

[Comfy Quants](https://github.com/Comfy-Org/comfy-quants) exports prequantized checkpoints with per-layer
quantization metadata. Diffusers reads each layer's format and uses
[Comfy-Kitchen](https://github.com/Comfy-Org/comfy-kitchen) for inference. A checkpoint can mix quantization
formats and floating-point layers.

Install the optional dependencies:

```bash
pip install "comfy-kitchen>=0.2.41" accelerate
```

## Load a checkpoint

Pass [`ComfyQuantizationConfig`] to `from_single_file`. Set `compute_dtype` to BF16 or FP16; the checkpoint
determines the quantization format of each layer.

```python
import torch
from diffusers import ComfyQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_single_file(
    "https://huggingface.co/Comfy-Org/Z-Image-Turbo/blob/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
    config="Tongyi-MAI/Z-Image-Turbo",
    subfolder="transformer",
    quantization_config=ComfyQuantizationConfig(compute_dtype=torch.bfloat16),
)
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = pipe(
    "A small orange cat sitting on a mossy stone wall in a mountain village",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

## Validated formats

The quantizer reads each layer's uint8 JSON `comfy_quant` marker and bundles the packed weight with its scales and other sidecar tensors before checkpoint conversion.


| Marker `format` | Stored weights | Associated tensors |
| --- | --- | --- |
| `float8_e4m3fn`, `float8_e5m2` | FP8 | Weight scale, optional activation scale |
| `int8_tensorwise` | INT8 | Scalar or row scales, optional ConvRot metadata |
| `mxfp8` | FP8 | E8M0 block scales |
| `convrot_w4a4` | Packed INT4 | Row scales and rotation metadata |
| `asym_w4a8_int8`, `w6a8_int8` | Packed INT4 or INT6 | Group and channel scales, optional codebook and correction |

Checkpoint converters retain their usual weight splitting and concatenation. The quantized tensor keeps row
and block scales aligned without dequantizing the stored weights. Concatenation requires compatible
layouts and shared metadata. Input-feature splitting is unsupported.

One `ComfyLinear` dispatches through the weight's Kitchen layout. Layers marked `full_precision_matrix_mult`
use dequantized matrix multiplication. Unquantized layers use the compute dtype, and model FP32 exclusions
receive dequantized FP32 weights.

Kitchen selects an available execution backend. Hardware requirements differ by format; loading a checkpoint
does not guarantee native quantized acceleration on the current device.

### INT8 ConvRot

[ConvRot](https://arxiv.org/abs/2512.03673) rotates weights offline and activations at inference time to spread
activation outliers. Native `int8_tensorwise` checkpoints retain INT8 weights and FP32 scales. Rotated layers
include `convrot` and `convrot_groupsize` metadata. Both rotated and unrotated INT8 layers are supported.
