<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# SDNQ

[SDNQ](https://github.com/Disty0/sdnq) (SD.Next Quantization) is a training-free quantization backend supporting a wide range of weight dtypes (int8 down to 2-bit, FP8 and other low-bit float formats) with optional SVD correction, Hadamard rotation, and quantized INT8/FP8 matmul for faster inference. It runs on CUDA, ROCm, XPU, MPS, and CPU.

Install the `sdnq` library to use this backend.

```bash
pip install sdnq
```

## Load a prequantized checkpoint

The quantization configuration of a prequantized SDNQ checkpoint is stored in its `config.json`, so it loads like any other model, with no `quantization_config` argument needed. The only requirement is having `sdnq` installed.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    dtype=torch.bfloat16,
).to("cuda")

image = pipe("a cat holding a sign that says hello").images[0]
image.save("output.png")
```

## Quantize a model on the fly

Pass an [`SDNQConfig`] to `from_pretrained` to quantize a model during loading. Use [`PipelineQuantizationConfig`] to quantize specific pipeline components.

```python
import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, SDNQConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": SDNQConfig(weights_dtype="uint4", use_svd=True)}
)
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    quantization_config=pipeline_quant_config,
    dtype=torch.bfloat16,
).to("cuda")
```

Or quantize a single model component directly.

```python
import torch
from diffusers import ZImageTransformer2DModel, SDNQConfig

transformer = ZImageTransformer2DModel.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    subfolder="transformer",
    quantization_config=SDNQConfig(weights_dtype="int8"),
    dtype=torch.bfloat16,
)
```

All arguments are forwarded to [`sdnq.SDNQConfig`](https://github.com/Disty0/sdnq). Refer to the SDNQ documentation for the full list of supported dtypes and options.

## Choosing which layers to quantize

Some layers are more sensitive to quantization than others, and skipping the sensitive ones improves quality at little memory cost. SDNQ handles this out of the box: it ships built-in skip lists (a generic one plus per-model lists for many architectures) that are applied automatically, so the usual sensitive layers stay unquantized without any configuration. For finer control it can also select layers dynamically, measuring the per-layer quantization error and skipping the ones above a loss threshold. Refer to the [SDNQ dynamic quantization guide](https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization#dynamic-loss-threshold) for details.

## Faster inference with quantized matmul

By default SDNQ dequantizes weights back to the compute dtype before each matmul. Enabling quantized matmul instead runs the matmul directly in a low-precision dtype (INT8/FP8/FP16), which is faster on hardware with a working [Triton](https://github.com/triton-lang/triton) install (CUDA, ROCm, or XPU).

Enable it on a loaded model with `apply_sdnq_options_to_model`, which reads the model's existing quantization config and re-applies the option in place. This works for prequantized checkpoints too, where no `SDNQConfig` is passed.

```python
import torch
from diffusers import DiffusionPipeline
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

pipe = DiffusionPipeline.from_pretrained(
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
    dtype=torch.bfloat16,
).to("cuda")

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
```

## torch.compile

SDNQ quantized models are compatible with [torch.compile](../optimization/fp16#torchcompile) for further speedups, including together with [group offloading](../optimization/memory#group-offloading).

```python
transformer.compile(fullgraph=True)
```

## Save a quantized model

Quantized models can be serialized with `save_pretrained` and reloaded without a `quantization_config`.

```python
transformer.save_pretrained("z-image-turbo-sdnq-int8")
```

## Resources

- [SDNQ repository](https://github.com/Disty0/sdnq)
- [Prequantized SDNQ models on the Hub](https://huggingface.co/models?search=sdnq)
