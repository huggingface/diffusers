<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# BFL (fp8r / nvfp4)

[Black Forest Labs](https://huggingface.co/black-forest-labs) ships quantized variants of its [Flux3](../api/pipelines/flux3) transformers in two schemes, stored as single-file checkpoints next to the original bf16 weights:

- `fp8r`: rowwise FP8. Each quantized linear stores an E4M3 `weight` and an fp32 per-row `weight_scale`. Activations are quantized to E4M3 per row at runtime.
- `nvfp4`: NVFP4 in the [ModelOpt](https://github.com/NVIDIA/Model-Optimizer) layout. Each quantized linear stores a packed E2M1 `weight` (two values per byte), E4M3 block scales of 16 (`weight_scale`), a global fp32 `weight_scale_2`, and a calibrated fp32 activation scale `input_scale`.

Diffusers loads both through [`~FromSingleFileMixin.from_single_file`] with a [`BFLQuantizationConfig`]. The scheme is read from the checkpoint, so the same config works for either file. Only loading prequantized checkpoints is supported; pipeline-level loading is not.

The example below loads the `fp8r` checkpoint of [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base), which lives at `variants/fp8r/model.safetensors` in the model repository. Pass `config` so that the model config is read from the Diffusers folder of that repository, then load the other components by name (see [Single-file loading](../api/pipelines/flux3#single-file-loading)).

```python
import torch

from diffusers import BFLQuantizationConfig, Flux3Transformer2DModel, ModularPipeline

ckpt_path = "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/fp8r/model.safetensors"
transformer = Flux3Transformer2DModel.from_single_file(
    ckpt_path,
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype=torch.bfloat16)
pipe.update_components(transformer=transformer)
pipe.to("cuda")

prompt = "A wide-angle wildlife photograph of a red fox standing alert in a snow-covered forest clearing at golden hour."
image = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
image.save("flux3-klein-base-fp8r.png")
```

Replace `variants/fp8r` with `variants/nvfp4` to load the `nvfp4` variant. The step-distilled [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) ships the same two variants; load them with its repository id as `config` and pipeline.

## Kernels

The quantized weights stay in their low-precision storage dtype. Which GEMM runs depends on the device:

| Scheme | Fused kernel | Requirements |
|---|---|---|
| `fp8r` | `torch._scaled_mm` with rowwise scales | CUDA GPU with compute capability 8.9 or higher |
| `nvfp4` | `flashinfer.mm_fp4` (CUTLASS) with `flashinfer.nvfp4_quantize` for the activations | Blackwell GPU (compute capability 10.0 or higher), a CUDA toolkit with `nvcc`, and `flashinfer-python` with `flashinfer-cubin` at the same version |

Install the two FlashInfer packages at the same version, for example `pip install flashinfer-python==0.6.13 flashinfer-cubin==0.6.13`. The `flashinfer-cubin` release can lag behind `flashinfer-python`, and FlashInfer refuses to import when the versions differ.

FlashInfer compiles its NVFP4 kernels on first use with the CUDA toolkit of the environment and picks the target architecture from the visible GPU (or from `FLASHINFER_CUDA_ARCH_LIST` when set). Consumer and workstation Blackwell GPUs (compute capability 12.0) need a CUDA toolkit of 12.9 or newer. With an older toolkit FlashInfer finds no eligible architecture and the first NVFP4 call fails with `FlashInfer requires GPUs with sm75 or higher`.

On any other device the weights are dequantized to `compute_dtype` in each forward pass and the layer runs as a regular linear. In that case the activations are not quantized, so outputs differ slightly from the fused kernels.
