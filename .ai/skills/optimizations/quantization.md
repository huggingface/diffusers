# Quantization

## Documentation

Read the full guides for usage examples per backend:

- **Local:** `docs/source/en/quantization/overview.md`, `bitsandbytes.md`, `torchao.md`, `gguf.md`
- **Online:** https://huggingface.co/docs/diffusers/main/en/quantization/overview

## Critical: use `PipelineQuantizationConfig`, not backend configs directly

**Pipeline-level quantization is the correct approach.** Pass a `PipelineQuantizationConfig` to `from_pretrained`. Passing a `BitsAndBytesConfig` directly raises:
> `quantization_config must be an instance of PipelineQuantizationConfig`

## Backend name strings

The `quant_backend` field requires exact strings — these differ from the config class names:

| `quant_backend` value | Notes |
|---|---|
| `"bitsandbytes_4bit"` | NOT `"bitsandbytes"` — suffix is required |
| `"bitsandbytes_8bit"` | NOT `"bitsandbytes"` — suffix is required |
| `"gguf"` | |
| `"modelopt"` | |

**torchao cannot use `quant_backend`.** It requires `quant_mapping` instead — see below.

## torchao: use `quant_mapping`, not `quant_backend`

torchao requires `TorchAoConfig` which takes an `AOBaseConfig` instance (not a string). The `quant_backend` path would try to construct `TorchAoConfig(quant_type="int8_weight_only")` — a string — which fails at `post_init()` with `TypeError: quant_type must be an AOBaseConfig instance`.

Always use `quant_mapping` for torchao:

```python
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig())}
)
pipe = DiffusionPipeline.from_pretrained(
    "model_id",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",  # torchao quantizes efficiently on CUDA — use "cpu" only if VRAM is tight
)
```

For the full list of `AOBaseConfig` classes, see `docs/source/en/quantization/torchao.md` / https://huggingface.co/docs/diffusers/main/en/quantization/torchao.

## `quant_kwargs` must be non-empty

Passing `quant_kwargs={}` or `quant_kwargs=None` raises:
> `ValueError: Both quant_kwargs and quant_mapping cannot be None`

Always pass at least one kwarg. For `bitsandbytes_8bit` (which has no required kwargs), use:
```python
quant_kwargs={"load_in_8bit": True}
```

## Always use `device_map="cpu"` when loading

Without it, full-precision weights load to GPU first (causing OOM), then get quantized. With `device_map="cpu"`, quantization happens on CPU and only the quantized weights ever touch VRAM.

After loading, move to GPU with either:
- `pipe.to("cuda")` — if all quantized components fit in VRAM simultaneously
- `pipe.enable_model_cpu_offload()` — if they don't all fit at once

## `bitsandbytes_8bit` + `enable_model_cpu_offload()` fails at inference

`LLM.int8()` can only execute on CUDA. When `enable_model_cpu_offload()` moves the int8 component back to CPU between steps, the matmul fails.

**Fix:** keep the int8 component on CUDA permanently and use group offloading for the rest:
```python
pipe.enable_model_cpu_offload()
pipe.transformer.to("cuda")  # keep int8 transformer on GPU permanently
```
Or switch to `bitsandbytes_4bit`, which supports device moves.

## Text encoder quantization

**Text encoder quantization is a first-class optimization, not an afterthought.** Many modern models have LLM-based text encoders as large as the transformer itself. Always include large text encoders in `components_to_quantize`. See `docs/source/en/quantization/overview.md` for usage examples.

Reference sizes (bf16):

| Model family | Text encoder | Size |
|---|---|---|
| FLUX.2 Klein | Qwen3 | ~9 GB |
| FLUX.1 | T5-XXL | ~10 GB |
| SD3 | T5-XXL + CLIP-L + CLIP-G | ~11 GB total |
| CogVideoX | T5-XXL | ~10 GB |

Newer models (FLUX.2 Klein, etc.) use a **single LLM-based text encoder** — check the pipeline's `text_encoder` attribute. Do not assume a CLIP + T5 dual-encoder layout.

## Memory checks before recommending quantization

- **RAM >= `S_largest_component_bf16`** — full-precision weights of the largest component to quantize must fit in RAM during loading
- **VRAM >= `S_total_after_quant` + A** (for `pipe.to("cuda")`) or **VRAM >= `S_max_after_quant` + A** (for model CPU offload)

See [memory-calculator.md](memory-calculator.md) for size estimation formulas.

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `quantization_config must be an instance of PipelineQuantizationConfig` | Passed `BitsAndBytesConfig` directly | Wrap in `PipelineQuantizationConfig` |
| `quant_backend not found` | Wrong backend name string | Use `bitsandbytes_4bit` / `bitsandbytes_8bit`, not `bitsandbytes` |
| `TypeError: quant_type must be an AOBaseConfig instance` | Used `quant_backend="torchao"` with a string `quant_type` | Switch to `quant_mapping={"component": TorchAoConfig(Int8WeightOnlyConfig())}` |
| `Both quant_kwargs and quant_mapping cannot be None` | Empty `quant_kwargs` | Always pass at least one kwarg |
| OOM during loading | Missing `device_map="cpu"` | Add `device_map="cpu"` to `from_pretrained` |
| OOM during `pipe.to("cuda")` | Quantized model still too large | Use `enable_model_cpu_offload()` instead |
| int8 matmul fails on CPU | `bitsandbytes_8bit` + `enable_model_cpu_offload()` | Keep int8 component on CUDA; switch to `bitsandbytes_4bit` for offloading |
