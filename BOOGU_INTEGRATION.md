# Boogu-Image Integration into Diffusers

This document describes how the standalone **Boogu-Image** model (originally in
`Boogu-Image/boogu`) has been merged into this `diffusers` fork, what was added,
and how to use or review it.

## Summary

Boogu-Image is an instruction-driven image generation and editing model. It pairs a
Qwen3-VL multimodal LLM (instruction encoder) with a single/double-stream transformer
denoiser and a flow-matching scheduler that uses training-aligned time shifting.

The integration moves Boogu's source into the diffusers package tree, rewrites the
`boogu.*` imports to diffusers-internal imports, and registers the new classes through
the normal diffusers lazy-import machinery so they are importable as first-class
diffusers citizens:

```python
from diffusers import BooguImageTransformer2DModel, PromptEmbedding
from diffusers.pipelines.boogu import BooguImagePipeline, BooguImageTurboPipeline
```

## What was added

### Models (`src/diffusers/models/`)

| File | Contents |
|---|---|
| `transformers/transformer_boogu.py` | `BooguImageTransformer2DModel`, `PromptEmbedding` |
| `transformers/block_lumina2.py` | Lumina2 building blocks (RMSNorm-zero, feed-forward, timestep/caption embedding). `swiglu` helper inlined here. |
| `transformers/rope_boogu.py` | Boogu rotary positional embeddings (`BooguImageRotaryPosEmbed`, double-stream / prompt-tuning variants) |
| `attention_processor_boogu.py` | Boogu attention processors (standard + flash-attn varlen, single/double-stream). Local `apply_rotary_emb` handles the Lumina-style (`use_real=False`) path safely for empty tensors. |

### Pipelines (`src/diffusers/pipelines/boogu/`)

| File | Contents |
|---|---|
| `pipeline_boogu.py` | `BooguImagePipeline` (text-to-image and instruction editing), `FMPipelineOutput` |
| `pipeline_boogu_turbo.py` | `BooguImageTurboPipeline` — DMD few-step T2I subclass. Defaults the guidance scales to the DMD-required values (`text=1.0`, `image=1.0`, `empty=0.0`). |
| `lora_pipeline.py` | `BooguImageLoraLoaderMixin` |
| `image_processor.py` | `BooguImageProcessor` |
| `instruct_reasoner_static_skills.py`, `static_skills.py` | Prompt-rewriting skill tables |

### Scheduler (`src/diffusers/schedulers/`)

`scheduling_flow_match_euler_discrete_time_shifting.py` — a flow-matching Euler scheduler
with Boogu's training-aligned time shift (`v1` logistic and `v2` rational variants,
static or dynamic). Class name is `FlowMatchEulerDiscreteScheduler`; import it via its
module path to avoid clashing with the built-in scheduler of the same name.

### Internal utilities

| Location | Contents |
|---|---|
| `src/diffusers/cache_functions/` | DPM / force-scheduler caching helpers |
| `src/diffusers/taylorseer_utils/` | TaylorSeer derivative-approximation inference cache |
| `src/diffusers/ops/triton/` | Optional Triton fused RMSNorm (falls back to `torch.nn.RMSNorm`) |
| `src/diffusers/utils/teacache_util.py` | `TeaCacheParams` |
| `src/diffusers/utils/validator_utils.py` | device / offload validation helpers |

### Changes to existing diffusers files

| File | Change |
|---|---|
| `src/diffusers/__init__.py` | Register `BooguImage*` model & pipeline names |
| `src/diffusers/models/__init__.py`, `models/transformers/__init__.py` | Register transformer + `PromptEmbedding` |
| `src/diffusers/pipelines/__init__.py` | Register `boogu` pipeline group |
| `src/diffusers/schedulers/__init__.py` | (Boogu scheduler loaded by module path; no top-level alias to avoid name clash) |
| `src/diffusers/utils/import_utils.py` | Add `is_triton_available()` |
| `src/diffusers/pipelines/pipeline_loading_utils.py` | Add `_DIFFUSERS_MODULE_ALIASES` (see below) |

## Loading published checkpoints without remote code

Boogu checkpoints ship a `model_index.json` whose `transformer` / `scheduler` entries
point at custom module names (e.g. `transformer_boogu`,
`boogu.models.transformers.transformer_boogu`,
`scheduling_flow_match_euler_discrete_time_shifting`). By default diffusers would try to
load these as remote/local custom code and require `trust_remote_code=True`.

To use the *integrated* classes instead, `pipeline_loading_utils.py` defines
`_DIFFUSERS_MODULE_ALIASES`, a small map from those custom module names to the
integrated diffusers modules. The loader consults it in three places
(`get_class_obj_and_candidates`, `maybe_raise_or_warn`,
`_get_custom_components_and_folders`), so `from_pretrained` resolves the published
config to the in-tree classes with **no config edits and no `trust_remote_code`**:

```python
from diffusers.pipelines.boogu import BooguImagePipeline

pipe = BooguImagePipeline.from_pretrained("Boogu/Boogu-Image-0.1-Base")
```

## Examples

Runnable inference scripts (base / turbo / edit, plus FP8 variants) and their own
README live in [`examples/boogu/`](examples/boogu/README.md).

## Optional performance dependencies

The transformer uses fused kernels when present, otherwise falls back to pure PyTorch
with a one-time warning:

- `triton` — fused RMSNorm
- `flash_attn` — fused SwiGLU and variable-length flash attention

## Notes for reviewers

- `block_lumina2.py` and `rope_boogu.py` are kept as separate files (the rope module is
  reused by both the transformer and the pipeline; `block_lumina2` keeps the already
  large `transformer_boogu.py` readable). The tiny `components.py` helper was inlined.
- `embeddings_boogu.py` was removed: its `apply_rotary_emb` is a subset of the shared
  `diffusers.models.embeddings.apply_rotary_emb`, and its `TimestepEmbedding` was unused.
- The Boogu scheduler intentionally keeps the upstream class name
  `FlowMatchEulerDiscreteScheduler`; it is distinguished by its module path. Promoting
  its `v2` time-shift formula into the upstream scheduler as a new `time_shift_type`
  would be a reasonable follow-up.
