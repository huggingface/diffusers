<!--Copyright 2025 - 2026 Advanced Micro Devices, Inc. and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quark

[Quark](https://quark.docs.amd.com/latest/) is AMD's deep-learning quantization toolkit. It is agnostic to specific data types, algorithms, and hardware, and primarily targets AMD CPUs and GPUs. Quark supports a broad range of strategies — INT8, INT4, FP8, MX, FP4, SVDQuant, SmoothQuant, AWQ, GPTQ, QuaRot, SpinQuant — combinable across diffusion submodules (UNet, transformer, VAE).

The Diffusers integration mirrors the [Transformers integration](https://huggingface.co/docs/transformers/quantization/quark): models exported with `quark.torch.export_safetensors` can be loaded back through `DiffusionPipeline.from_pretrained` / `ModelMixin.from_pretrained` without per-layer setup code.

To use Quark with Diffusers, install Quark:

```bash
pip install amd-quark
```

## Loading a pre-quantized model

If a model on the Hub already carries a `quantization_config` block in `config.json`, no extra setup is needed:

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "amd/sd3-quark-int8",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("A cat on a windowsill", num_inference_steps=30).images[0]
```

The dispatch is automatic: the loader sees `quant_method = "quark"` and instantiates `QuarkDiffusersQuantizer`.

## On-the-fly weight-only quantization

Pass `QuarkConfig(...)` against a vanilla fp16/bf16 model to quantize weights at load time:

```python
import torch
from diffusers import StableDiffusion3Pipeline, QuarkConfig

# A QConfig that produces INT8 weight-only quantization (no activation quantizers).
# Build with quark.torch.quantization.config.config.QConfig and pass its dict.
quark_config_dict = ...  # see https://quark.docs.amd.com/latest/

quantization_config = QuarkConfig(quant_method="quark", **quark_config_dict)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
).to("cuda")
```

This works for any QConfig that does not declare activation quantizers (input or output `QTensorConfig`). Examples: INT8 weight-only, MXFP4 weight-only.

For activation-quantized configurations (SmoothQuant, SVDQuant w4a4, FP8 with calibrated activations, etc.), `from_pretrained` will raise a `NotImplementedError` directing you to the offline path.

## Producing a quantized checkpoint

For configurations that need calibration data, use the offline workflow:

```python
import torch
from diffusers import StableDiffusion3Pipeline
from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.utils.diffusers import get_calib_dataloader

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
).to("cuda")

prompts = [
    "A serene lake reflecting mountains at sunset",
    "A futuristic city with flying cars at night",
]
dataloader = get_calib_dataloader(pipe, pipe.transformer, prompts, n_steps=20)

qconfig = ...  # SVDQuant / SmoothQuant / FP8 + activation calibration
pipe.transformer = ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)

export_safetensors(pipe.transformer, "sd3-quark-svdquant/transformer")
```

The exported directory then reloads through `from_pretrained` per the first section.

## Support matrix

| Feature | Supported |
| --- | --- |
| Data types | INT8, INT4, INT2, BFloat16, Float16, FP8 (E4M3/E5M2), FP6, FP4, OCP MX, MX6, MX9, BFP16 |
| Pre-quantization transforms | SmoothQuant, QuaRot, SpinQuant, AWQ |
| Quantization algorithms | GPTQ, SVDQuant |
| Operators | `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.Embedding`, `nn.EmbeddingBag` |
| Granularity | per-tensor, per-channel, per-group, per-block, per-layer, per-layer-type |
| Activation calibration | min/max, percentile, histogram, MSE |
| Quantization strategy | weight-only, static, dynamic, with or without output quantization |
| `torch.compile` | yes (after `ModelQuantizer.freeze`) |

## Resources

- Quark documentation: <https://quark.docs.amd.com/latest/>
- Quark-quantized models on the Hub: <https://huggingface.co/models?other=quark>
