<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# GGUF

The GGUF file format is typically used to store models for inference with [GGML](https://github.com/ggerganov/ggml) and supports a variety of block wise quantization options. Diffusers supports loading checkpoints prequantized and saved in the GGUF format via `from_single_file` loading with Model classes. Loading GGUF checkpoints via Pipelines is currently not supported.

The following example will load the [FLUX.1 DEV](https://huggingface.co/black-forest-labs/FLUX.1-dev) transformer model using the GGUF Q2_K quantization variant.

Before starting please install gguf in your environment

```shell
pip install -U gguf
```

Since GGUF is a single file format, use [`~FromSingleFileMixin.from_single_file`] to load the model and pass in the [`GGUFQuantizationConfig`].

When using GGUF checkpoints, the quantized weights remain in a low memory `dtype`(typically `torch.uint8`) and are dynamically dequantized and cast to the configured `compute_dtype` during each module's forward pass through the model. The `GGUFQuantizationConfig` allows you to set the `compute_dtype`.

The functions used for dynamic dequantizatation are based on the great work done by [city96](https://github.com/city96/ComfyUI-GGUF), who created the Pytorch ports of the original [`numpy`](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/quants.py) implementation by [compilade](https://github.com/compilade).

```python
import torch

from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

ckpt_path = (
    "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
)
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
image.save("flux-gguf.png")
```

## Supported Quantization Types

- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q5_K
- Q6_K

