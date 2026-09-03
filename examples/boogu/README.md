# Boogu-Image

[Boogu-Image](https://huggingface.co/Boogu) is an instruction-driven image generation and editing model. It pairs a Qwen3-VL multimodal LLM (instruction encoder) with a single/double-stream transformer denoiser and a flow-matching scheduler with training-aligned time shifting.

This directory contains minimal inference scripts for the released checkpoints.

## Environment installation
[Boogu-Image-quick-start](https://github.com/boogu-project/Boogu-Image/blob/main/quick_start.sh)

## Pipelines

| Pipeline | Class | Use case |
|---|---|---|
| Base | `BooguImagePipeline` | Text-to-image (50 steps) |
| Turbo | `BooguImageTurboPipeline` | Few-step DMD text-to-image (4 steps) |
| Edit | `BooguImagePipeline` | Instruction-based image editing (pass `input_images`) |

## Scripts

| Script | Checkpoint |
|---|---|
| `inference_base.py` | `Boogu/Boogu-Image-0.1-Base` |
| `inference_turbo.py` | `Boogu/Boogu-Image-0.1-Turbo` |
| `inference_edit.py` | `Boogu/Boogu-Image-0.1-Edit` |
| `inference_base_fp8.py` | `Boogu/Boogu-Image-0.1-Base-fp8` |
| `inference_turbo_fp8.py` | `Boogu/Boogu-Image-0.1-Turbo-fp8` |
| `inference_edit_fp8.py` | `Boogu/Boogu-Image-0.1-Edit-fp8` |

## Usage

Text-to-image:

```bash
python inference_base.py
```

Few-step (Turbo):

```bash
python inference_turbo.py
```

Image editing (reads `base.png` as the reference image, so run `inference_base.py` first):

```bash
python inference_edit.py
```

## FP8 checkpoints

FP8 weights are stored in a non-safetensors format, so the transformer is loaded
separately with `use_safetensors=False` and passed to the pipeline:

```python
import torch
from diffusers import BooguImageTransformer2DModel
from diffusers.pipelines.boogu import BooguImagePipeline

transformer = BooguImageTransformer2DModel.from_pretrained(
    "Boogu/Boogu-Image-0.1-Base-fp8",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=False,
)
pipe = BooguImagePipeline.from_pretrained(
    "Boogu/Boogu-Image-0.1-Base-fp8", torch_dtype=torch.bfloat16, transformer=transformer
)
pipe = pipe.to("cuda")
```

The FP8 scripts also disable the DeepGEMM kernel for the FP8 VLM (forcing a Triton
finegrained-fp8 fallback) for broader hardware compatibility — see
`_disable_deepgemm_for_fp8_vlm()` in each FP8 script.

## Optional performance dependencies

The transformer can use fused kernels when available; without them it falls back to
pure PyTorch and prints a one-time warning:

- `triton` — fused RMSNorm
- `flash_attn` — fused SwiGLU and variable-length flash attention
