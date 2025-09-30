<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# GGUF

GGUF is a binary file format for storing and loading [GGML](https://github.com/ggerganov/ggml) models for inference. It's designed to support various blockwise quantization options, single-file deployment, and fast loading and saving.

Diffusers only supports loading GGUF *model* files as opposed to an entire GGUF pipeline checkpoint.

<details>
<summary>Supported quantization types</summary>

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

</details>

Make sure gguf is installed.

```bash
pip install -U gguf
```

Load GGUF files with [`~loaders.FromSingleFileMixin.from_single_file`] and pass [`GGUFQuantizationConfig`] to configure the `compute_type`. Quantized weights remain in a low memory data type and are dynamically dequantized and cast to the configured `compute_dtype` during each module's forward pass through the model.

```python
import torch
from diffusers import FluxPipeline, AutoModel, GGUFQuantizationConfig

transformer = AutoModel.from_single_file(
    "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline(prompt).images[0]
image.save("flux-gguf.png")
```

## CUDA kernels

Optimized CUDA kernels accelerate GGUF model inference by ~10%. You need a compatible GPU with `torch.cuda.get_device_capability` greater than 7 and the [kernels](https://huggingface.co/docs/kernels/index) library.

```bash
pip install -U kernels
```

Set `DIFFUSERS_GGUF_CUDA_KERNELS=true` to enable optimized kernels. CUDA kernels introduce minor numerical differences compared to the original GGUF implementation, which may cause subtle visual variations in generated images.

```python
import os

# Enable CUDA kernels for ~10% speedup
os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "true"
# Disable CUDA kernels
# os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "false"
```

## Convert to GGUF

Use the Space below to convert a Diffusers checkpoint into a GGUF file.

<iframe
	src="https://diffusers-internal-dev-diffusers-to-gguf.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

GGUF files stored in the [Diffusers format](../using-diffusers/other-formats) require the model's `config` path. Provide the `subfolder` argument if the model config is inside a subfolder.

```py
import torch
from diffusers import FluxPipeline, AutoModel, GGUFQuantizationConfig

transformer = AutoModel.from_single_file(
    "https://huggingface.co/sayakpaul/different-lora-from-civitai/blob/main/flux_dev_diffusers-q4_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    config="black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```