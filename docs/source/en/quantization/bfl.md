<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# BFL (fp8r / nvfp4)

[Black Forest Labs](https://huggingface.co/black-forest-labs) ships quantized variants of its transformers in two schemes, stored as single-file checkpoints next to the original bf16 weights:

- `fp8r`: rowwise FP8. Each quantized linear stores an E4M3 `weight` and an fp32 per-row `weight_scale`. Activations are quantized to E4M3 per row at runtime.
- `nvfp4`: NVFP4 in the [ModelOpt](https://github.com/NVIDIA/Model-Optimizer) layout. Each quantized linear stores a packed E2M1 `weight` (two values per byte), E4M3 block scales of 16 (`weight_scale`), a global fp32 `weight_scale_2`, and a calibrated fp32 activation scale `input_scale`.

Diffusers loads both through [`~FromSingleFileMixin.from_single_file`] with a [`BFLQuantizationConfig`]. The scheme is read from the checkpoint, so the same config works for either file. Only loading prequantized checkpoints is supported; pipeline-level loading is not.

The example below loads a `fp8r` checkpoint of [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Pass `config` so that the model config is read from the Diffusers repository.

```python
import torch

from diffusers import BFLQuantizationConfig, Flux2KleinPipeline, Flux2Transformer2DModel

ckpt_path = "https://huggingface.co/sayakpaul/FLUX.2-klein-4B-fp8r-nvfp4/blob/main/flux-2-klein-4b-fp8r.safetensors"
transformer = Flux2Transformer2DModel.from_single_file(
    ckpt_path,
    config="black-forest-labs/FLUX.2-klein-4B",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    transformer=transformer,
    dtype=torch.bfloat16,
).to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.0, generator=torch.manual_seed(0)).images[0]
image.save("flux2-klein-fp8r.png")
```

Replace the filename with `flux-2-klein-4b-nvfp4.safetensors` to load the `nvfp4` variant.

Both files were produced with [`scripts/quantize_flux2_bfl.py`](https://github.com/huggingface/diffusers/blob/main/scripts/quantize_flux2_bfl.py), which quantizes the block linears of a FLUX.2 single-file checkpoint and, for `nvfp4`, calibrates the activation scales on a few prompts.

## Kernels

The quantized weights stay in their low-precision storage dtype. Which GEMM runs depends on the device:

| Scheme | Fused kernel | Requirements |
|---|---|---|
| `fp8r` | `torch._scaled_mm` with rowwise scales | CUDA GPU with compute capability 8.9 or higher |
| `nvfp4` | `flashinfer.mm_fp4` (CUTLASS) with `flashinfer.nvfp4_quantize` for the activations | Blackwell GPU (compute capability 10.0 or higher) and `pip install flashinfer-python flashinfer-cubin` |

FlashInfer compiles its NVFP4 kernels on first use and reads the target architecture from `TORCH_CUDA_ARCH_LIST`. On consumer and workstation Blackwell GPUs (compute capability 12.0) set `TORCH_CUDA_ARCH_LIST=12.0a` before running, otherwise FlashInfer refuses the device.

On any other device the weights are dequantized to `compute_dtype` in each forward pass and the layer runs as a regular linear. In that case the activations are not quantized, so outputs differ slightly from the fused kernels.
