# Quantization

## Overview

Quantization reduces model weights from fp16/bf16 to lower precision (int8, int4, fp8), cutting memory usage and often improving throughput. Diffusers supports several quantization backends.

## Supported backends

| Backend | Precisions | Key features |
|---|---|---|
| **bitsandbytes** | int8, int4 (nf4/fp4) | Easiest to use, widely supported, QLoRA training |
| **torchao** | int8, int4, fp8 | PyTorch-native, good for inference, `autoquant` support |
| **GGUF** | Various (Q4_K_M, Q5_K_S, etc.) | Load GGUF checkpoints directly, community quantized models |

## Critical: Pipeline-level vs component-level quantization

**Pipeline-level quantization is the correct approach.** Pass a `PipelineQuantizationConfig` to `from_pretrained`. Do NOT pass a `BitsAndBytesConfig` directly — the pipeline's `from_pretrained` will reject it with `"quantization_config must be an instance of PipelineQuantizationConfig"`.

### Backend names in `PipelineQuantizationConfig`

The `quant_backend` string must match one of the registered backend keys. These are NOT the same as the config class names:

| `quant_backend` value | Notes |
|---|---|
| `"bitsandbytes_4bit"` | NOT `"bitsandbytes"` — the `_4bit` suffix is required |
| `"bitsandbytes_8bit"` | NOT `"bitsandbytes"` — the `_8bit` suffix is required |
| `"gguf"` | |
| `"torchao"` | |
| `"modelopt"` | |

### `quant_kwargs` for bitsandbytes

**`quant_kwargs` must be non-empty.** The validator raises `ValueError: Both quant_kwargs and quant_mapping cannot be None` if it's `{}` or `None`. Always pass at least one kwarg.

For `bitsandbytes_4bit`, the quantizer class is selected by backend name — `load_in_4bit=True` is redundant (the quantizer ignores it) but harmless. Pass the bnb-specific options instead:

```python
quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"}
```

For `bitsandbytes_8bit`, there are no bnb_8bit-specific kwargs, so pass the flag explicitly to satisfy the non-empty requirement:

```python
quant_kwargs={"load_in_8bit": True}
```

## Usage patterns

### bitsandbytes (pipeline-level, recommended)

```python
from diffusers import PipelineQuantizationConfig, DiffusionPipeline

quantization_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
    components_to_quantize=["transformer"],  # specify which components to quantize
)

pipe = DiffusionPipeline.from_pretrained(
    "model_id",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # load on CPU first to avoid OOM during quantization
)
```

### torchao (pipeline-level)

```python
from diffusers import PipelineQuantizationConfig, DiffusionPipeline

quantization_config = PipelineQuantizationConfig(
    quant_backend="torchao",
    quant_kwargs={"quant_type": "int8_weight_only"},
    components_to_quantize=["transformer"],
)

pipe = DiffusionPipeline.from_pretrained(
    "model_id",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
```

### GGUF (pipeline-level)

```python
from diffusers import PipelineQuantizationConfig, DiffusionPipeline

quantization_config = PipelineQuantizationConfig(
    quant_backend="gguf",
    quant_kwargs={"compute_dtype": torch.bfloat16},
)

pipe = DiffusionPipeline.from_pretrained(
    "model_id",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
```

## Loading: memory requirements and `device_map="cpu"`

Quantization is NOT free at load time. The full-precision (bf16/fp16) weights must be loaded into memory first, then compressed. This means:

- **Without `device_map="cpu"`** (default): each component loads to GPU in full precision, gets quantized on GPU, then the full-precision copy is freed. But while loading, you need VRAM for the full-precision weights of the current component PLUS all previously loaded components (already quantized or not). For large models, this causes OOM.
- **With `device_map="cpu"`**: components load and quantize on CPU. This requires **RAM >= S_component_bf16** for the largest component being quantized (the full-precision weights must fit in RAM during quantization). After quantization, RAM usage drops to the quantized size.

**Always pass `device_map="cpu"` when using quantization.** Then choose how to move to GPU:

1. **`pipe.to(device)`** — moves everything to GPU at once. Only works if all components (quantized + non-quantized) fit in VRAM simultaneously: `VRAM >= S_total_after_quant`.
2. **`pipe.enable_model_cpu_offload(device=device)`** — moves components to GPU one at a time during inference. Use this when `S_total_after_quant > VRAM` but `S_max_after_quant + A <= VRAM`.

### Memory check before recommending quantization

Before recommending quantization, verify:
- **RAM >= S_largest_component_bf16** — the full-precision weights of the largest component to be quantized must fit in RAM during loading
- **VRAM >= S_total_after_quant + A** (for `pipe.to()`) or **VRAM >= S_max_after_quant + A** (for model CPU offload) — the quantized model must fit during inference

## `components_to_quantize`

Use this parameter to control which pipeline components get quantized. Common choices:

- `["transformer"]` — quantize only the denoising model
- `["transformer", "text_encoder"]` — also quantize the text encoder (see below)
- `["transformer", "text_encoder", "text_encoder_2"]` — for dual-encoder models (FLUX.1, SD3, etc.) when both encoders are large
- Omit the parameter to quantize all compatible components

The VAE and vocoder are typically small enough that quantizing them gives little benefit and can hurt quality.

### Text encoder quantization

**Quantizing the text encoder is a first-class optimization, not an afterthought.** Many modern models use LLM-based text encoders that are as large as or larger than the transformer itself:

| Model family | Text encoder | Size (bf16) |
|---|---|---|
| FLUX.2 Klein | Qwen3 | ~9 GB |
| FLUX.1 | T5-XXL | ~10 GB |
| SD3 | T5-XXL + CLIP-L + CLIP-G | ~11 GB total |
| CogVideoX | T5-XXL | ~10 GB |

Newer models (FLUX.2 Klein, etc.) use a **single LLM-based text encoder** — check the pipeline definition for `text_encoder` vs `text_encoder_2`. Never assume CLIP+T5 dual-encoder layout.

When the text encoder is LLM-based, always include it in `components_to_quantize`. The combined savings often allow both components to fit in VRAM simultaneously, eliminating the need for CPU offloading entirely:

```python
# Both transformer (~4.5 GB) + Qwen3 text encoder (~4.5 GB) fit in VRAM at int4
quantization_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_quant_type": "nf4"},
    components_to_quantize=["transformer", "text_encoder"],
)
pipe = DiffusionPipeline.from_pretrained("model_id", quantization_config=quantization_config, device_map="cpu")
pipe.to("cuda")  # everything fits — no offloading needed
```

vs. transformer-only quantization, which may still require offloading because the text encoder alone exceeds available VRAM.

## Choosing a backend

- **Just want it to work**: bitsandbytes nf4 (`bitsandbytes_4bit`)
- **Best inference speed**: torchao int8 or fp8 (on supported hardware)
- **Using community GGUF files**: GGUF
- **Need to fine-tune**: bitsandbytes (QLoRA support)

## Common issues

- **OOM during loading**: You forgot `device_map="cpu"`. See the loading section above.
- **`quantization_config must be an instance of PipelineQuantizationConfig`**: You passed a `BitsAndBytesConfig` directly. Wrap it in `PipelineQuantizationConfig` instead.
- **`quant_backend not found`**: The backend name is wrong. Use `bitsandbytes_4bit` or `bitsandbytes_8bit`, not `bitsandbytes`. See the backend names table above.
- **`Both quant_kwargs and quant_mapping cannot be None`**: `quant_kwargs` is empty or `None`. Always pass at least one kwarg — see the `quant_kwargs` section above.
- **OOM during `pipe.to(device)` after loading**: Even quantized, all components don't fit in VRAM at once. Use `enable_model_cpu_offload()` instead of `pipe.to(device)`.
- **`bitsandbytes_8bit` + `enable_model_cpu_offload()` fails at inference**: `LLM.int8()` (bitsandbytes 8-bit) can only execute on CUDA — it cannot run on CPU. When `enable_model_cpu_offload()` moves the quantized component back to CPU between steps, the int8 matmul fails. **Fix**: keep the int8 component on CUDA permanently (`pipe.transformer.to("cuda")`) and use group offloading with `exclude_modules=["transformer"]` for the rest, or switch to `bitsandbytes_4bit` which supports device moves.
- **Quality degradation**: int4 can produce noticeable artifacts for some models. Try int8 first, then drop to int4 if memory requires it.
- **Slow first inference**: Some backends (torchao) compile/calibrate on first run. Subsequent runs are faster.
- **Incompatible layers**: Not all layer types support all quantization schemes. Check backend docs for supported module types.
- **Training**: Only bitsandbytes supports training (via QLoRA). Other backends are inference-only.
