<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# INT8 ConvRot

[ConvRot](https://arxiv.org/abs/2512.03673) uses group-wise regular Hadamard rotations to spread activation outliers.
Native Comfy INT8 checkpoints store rotated INT8 weights, FP32 scales, and per-layer rotation metadata. At inference,
[Comfy-Kitchen](https://github.com/Comfy-Org/comfy-kitchen) rotates and dynamically quantizes activations before INT8
matrix multiplication. Weights stay in INT8 throughout inference.

Install the optional runtime dependency:

```bash
pip install "comfy-kitchen>=0.2.41" accelerate
```

The native CUDA backend requires a CUDA 13.0 or newer PyTorch build and an NVIDIA GPU with INT8 tensor cores
(compute capability 7.5 or newer). Comfy-Kitchen selects the available backend; unsupported configurations may use
its eager backend. Memory savings do not guarantee faster inference.

## Load a checkpoint

Use [`ConvRotQuantizationConfig`] with `from_single_file` to load an existing native Comfy INT8 checkpoint.
Z-Image Turbo, Qwen-Image 2.1, and pruned or unpruned MiniMax-H3 FL2VA and REF2VA transformers are supported.

The INT8 linear runtime is shared across architectures. Loading another model also requires a compatible
checkpoint converter that maps its weights, scales, and rotation metadata, and a model implementation that uses
the floating compute dtype for activations.

```python
import torch
from diffusers import ConvRotQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel

transformer = ZImageTransformer2DModel.from_single_file(
    "https://huggingface.co/Comfy-Org/Z-Image-Turbo/blob/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
    config="Tongyi-MAI/Z-Image-Turbo",
    subfolder="transformer",
    quantization_config=ConvRotQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
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

For Qwen-Image 2.1, use `QwenImage21Transformer2DModel` with
`Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_int8_convrot.safetensors` and the configuration from
`Qwen/Qwen-Image-2.1`. Pass the loaded transformer to `QwenImage21Pipeline`.

To reduce host memory as well as VRAM peaks, encode prompts in a separate process, save the embeddings, and run
generation with `text_encoder=None`. Supply the embeddings and their masks on the execution device. Keep the
VAE offloaded during denoising and enable VAE tiling when needed.

## Supported format

Each quantized linear layer must have an INT8 `[out_features, in_features]` weight, an FP32 scalar or per-output-row
`weight_scale`, and a uint8 JSON `comfy_quant` tensor declaring `"format": "int8_tensorwise"`. Rotated layers require
per-output-row scales, `"convrot": true`, and a power-of-four `convrot_groupsize` that divides the input width.

The loader splits scales alongside fused QKV and gate/up weights, and preserves the associated rotation metadata.
Unquantized layers use the requested precision, except for modules the model keeps in FP32. BF16 and FP16
computation are supported.

This integration loads prequantized single-file transformer checkpoints. It does not support the older
ComfyUI-INT8-Fast marker format, other Comfy quantization formats, on-load quantization, training, LoRA, attention
projection fusion, or `save_pretrained` for the quantized transformer.

## MiniMax-H3

Load the pruned or unpruned Comfy transformer with the same config. The loader detects a pruned checkpoint's
`adaln_t_table`, infers its curve grid and rank, and retains the table and curve projections in FP32. Quantized
unpruned AdaLN projections use their floating compute dtype for activations.

```python
from diffusers import ConvRotQuantizationConfig, MiniMaxH3Transformer3DModel

transformer = MiniMaxH3Transformer3DModel.from_single_file(
    "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    config="MiniMaxAI/MiniMax-H3",
    subfolder="transformer",
    quantization_config=ConvRotQuantizationConfig(),
)
```

For the unpruned FL2VA model, select `minimax_h3_fl2va_int8_convrot.safetensors`. QKV splits and SwiGLU gate/value
reordering apply to the weights and their per-row scales together. The published Comfy files use the default
stacked QKV layout; a file merged directly from the original interleaved shards requires `qkv_layout="interleaved"`.

For REF2VA, select `minimax_h3_ref2va_pruned_int8_convrot.safetensors` or
`minimax_h3_ref2va_int8_convrot.safetensors`, load the configuration from `subfolder="transformer_ref"`, and pass
the loaded model as the modular pipeline's `transformer_ref` component.

MiniMax-H3 uses the existing modular video/audio workflow. Its INT8 transformer is still large, so phase prompt
encoding, denoising, and VAE decoding separately when host memory is limited, and offload transformer blocks as
needed. See the [MiniMax-H3 pipeline guide](../api/pipelines/minimax_h3) for packing and sampling constraints.
