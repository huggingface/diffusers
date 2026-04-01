# Memory Calculator

Use this guide to measure VRAM and RAM requirements for each optimization strategy, then recommend the best fit for the user's hardware.

## Step 1: Measure model sizes

**Do NOT guess sizes from parameter counts or model cards.** Pipelines often contain components that are not obvious from the model name (e.g., a pipeline marketed as having a "28B transformer" may also include a 24 GB text encoder, 6 GB connectors module, etc.). Always measure by running this snippet after loading the pipeline:

```python
import torch
from diffusers import DiffusionPipeline  # or the specific pipeline class

pipe = DiffusionPipeline.from_pretrained("model_id", torch_dtype=torch.bfloat16)

for name, component in pipe.components.items():
    if hasattr(component, 'parameters'):
        size_gb = sum(p.numel() * p.element_size() for p in component.parameters()) / 1e9
        print(f"{name}: {size_gb:.2f} GB")
```

For the transformer, also measure block-level and leaf-level sizes:

```python
# S_block: size of one transformer block
transformer = pipe.transformer
block_attr = None
for attr in ["transformer_blocks", "blocks", "layers"]:
    if hasattr(transformer, attr):
        block_attr = attr
        break
if block_attr:
    blocks = getattr(transformer, block_attr)
    block_size = sum(p.numel() * p.element_size() for p in blocks[0].parameters()) / 1e9
    print(f"S_block: {block_size:.2f} GB ({len(blocks)} blocks)")

# S_leaf: largest leaf module
max_leaf = max(
    (sum(p.numel() * p.element_size() for p in m.parameters(recurse=False))
     for m in transformer.modules() if list(m.parameters(recurse=False))),
    default=0
) / 1e9
print(f"S_leaf: {max_leaf:.4f} GB")
```

To measure the effect of layerwise casting on a component, apply it and re-measure:

```python
pipe.transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16,
)
size_after = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters()) / 1e9
print(f"Transformer after layerwise casting: {size_after:.2f} GB")
```

From the measurements, record:
- `S_total` = sum of all component sizes
- `S_max` = size of the largest single component
- `S_block` = size of one transformer block
- `S_leaf` = size of the largest leaf module
- `S_total_lc` = S_total after applying layerwise casting to castable components (measured, not estimated — norm/embed layers are skipped so it's not exactly half)
- `S_max_lc` = size of the largest component after layerwise casting (measured)
- `A` = activation memory during forward pass (cannot be measured ahead of time — estimate conservatively):
  - **Video models**: `A` scales with resolution and number of frames. A 5-second 960x544 video at 24fps can use ~7-8 GB. Higher resolution or more seconds = more activation memory.
  - **Image models**: `A` scales with image resolution. A 1024x1024 image might use 2-4 GB, but 2048x2048 could use 8-16 GB.
  - **Edit/inpainting models**: `A` includes the reference image(s) in addition to the generation activations, so budget extra.
  - When in doubt, estimate conservatively: `A ≈ 5-8 GB` for typical video workloads, `A ≈ 2-4 GB` for typical image workloads. For high-resolution or long video, increase accordingly.

## Step 2: Compute VRAM and RAM per strategy

### No optimization (all on GPU)

| | Estimate |
|---|---|
| **VRAM** | `S_total + A` |
| **RAM** | Minimal (just for loading) |
| **Speed** | Fastest — no transfers |
| **Quality** | Full precision |

### Model CPU offloading

| | Estimate |
|---|---|
| **VRAM** | `S_max + A` (only one component on GPU at a time) |
| **RAM** | `S_total` (all components stored on CPU) |
| **Speed** | Moderate — full model transfers between CPU/GPU per step |
| **Quality** | Full precision |

### Group offloading: block_level (no stream)

| | Estimate |
|---|---|
| **VRAM** | `num_blocks_per_group * S_block + A` |
| **RAM** | `S_total` (all weights on CPU, no pinned copy) |
| **Speed** | Moderate — synchronous transfers per group |
| **Quality** | Full precision |

Tune `num_blocks_per_group` to fill available VRAM: `floor((VRAM - A) / S_block)`.

### Group offloading: block_level (with stream)

Streams force `num_blocks_per_group=1`. Prefetches the next block while the current one runs.

| | Estimate |
|---|---|
| **VRAM** | `2 * S_block + A` (current block + prefetched next block) |
| **RAM** | `~2.5-3 * S_total` (original weights + pinned copies + allocation overhead) |
| **Speed** | Fast — overlaps transfer and compute |
| **Quality** | Full precision |

With `low_cpu_mem_usage=True`: RAM drops to `~S_total` (pins tensors on-the-fly instead of pre-pinning), but slower.

With `record_stream=True`: slightly more VRAM (delays memory reclamation), slightly faster (avoids stream synchronization).

> **Note on RAM estimates with streams:** Measured RAM usage is consistently higher than the theoretical `2 * S_total`. Pinned memory allocation, CUDA runtime overhead, and memory fragmentation add ~30-50% on top. Always use `~2.5-3 * S_total` when checking if the user has enough RAM for streamed offloading.

### Group offloading: leaf_level (no stream)

| | Estimate |
|---|---|
| **VRAM** | `S_leaf + A` (single leaf module, typically very small) |
| **RAM** | `S_total` |
| **Speed** | Slow — synchronous transfer per leaf module (many transfers) |
| **Quality** | Full precision |

### Group offloading: leaf_level (with stream)

| | Estimate |
|---|---|
| **VRAM** | `2 * S_leaf + A` (current + prefetched leaf) |
| **RAM** | `~2.5-3 * S_total` (pinned copies + overhead — see note above) |
| **Speed** | Medium-fast — overlaps transfer/compute at leaf granularity |
| **Quality** | Full precision |

With `low_cpu_mem_usage=True`: RAM drops to `~S_total`, but slower.

### Sequential CPU offloading (legacy)

| | Estimate |
|---|---|
| **VRAM** | `S_leaf + A` (similar to leaf_level group offloading) |
| **RAM** | `S_total` |
| **Speed** | Very slow — no stream support, synchronous per-leaf |
| **Quality** | Full precision |

Group offloading `leaf_level + use_stream=True` is strictly better. Prefer that.

### Layerwise casting (fp8 storage)

Reduces weight memory by casting to fp8. Norm and embedding layers are automatically skipped, so the reduction is less than 50% — always measure with the snippet above.

**`pipe.to()` caveat:** `pipe.to(device)` internally calls `module.to(device, dtype)` where dtype is `None` when not explicitly passed. This preserves fp8 weights. However, if the user passes dtype explicitly (e.g., `pipe.to("cuda", torch.bfloat16)` or the pipeline has internal dtype overrides), the fp8 storage will be overridden back to bf16. When in doubt, combine with `enable_model_cpu_offload()` which safely moves one component at a time without dtype overrides.

**Case 1: Everything on GPU** (if `S_total_lc + A <= VRAM`)

| | Estimate |
|---|---|
| **VRAM** | `S_total_lc + A` (measured — use the layerwise casting measurement snippet) |
| **RAM** | Minimal |
| **Speed** | Near-native — small cast overhead per layer |
| **Quality** | Slight degradation (fp8 weights, norm layers kept full precision) |

Use `pipe.to("cuda")` (without explicit dtype) after applying layerwise casting. Or move each component individually.

**Case 2: With model CPU offloading** (if Case 1 doesn't fit but `S_max_lc + A <= VRAM`)

| | Estimate |
|---|---|
| **VRAM** | `S_max_lc + A` (largest component after layerwise casting, one on GPU at a time) |
| **RAM** | `S_total` (all components on CPU) |
| **Speed** | Fast — small cast overhead per layer, component transfer overhead between steps |
| **Quality** | Slight degradation (fp8 weights, norm layers kept full precision) |

Apply layerwise casting to target components, then call `pipe.enable_model_cpu_offload()`.

### Layerwise casting + group offloading

Combines reduced weight size with offloading. The offloaded weights are in fp8, so transfers are faster and pinned copies smaller.

| | Estimate |
|---|---|
| **VRAM** | `num_blocks_per_group * S_block * 0.5 + A` (block_level) or `S_leaf * 0.5 + A` (leaf_level) |
| **RAM** | `S_total * 0.5` (no stream) or `~S_total` (with stream, pinned copy of fp8 weights) |
| **Speed** | Good — smaller transfers due to fp8 |
| **Quality** | Slight degradation from fp8 |

### Quantization (int4/nf4)

Quantization reduces weight memory but requires full-precision weights during loading. Always use `device_map="cpu"` so quantization happens on CPU.

Notation:
- `S_component_q` = quantized size of a component (int4/nf4 ≈ `S_component * 0.25`, int8 ≈ `S_component * 0.5`)
- `S_total_q` = total pipeline size after quantizing selected components
- `S_max_q` = size of the largest single component after quantization

**Loading (with `device_map="cpu"`):**

| | Estimate |
|---|---|
| **RAM (peak during loading)** | `S_largest_component_bf16` — full-precision weights of the largest component must fit in RAM during quantization |
| **RAM (after loading)** | `S_total_q` — all components at their final (quantized or bf16) sizes |

**Inference with `pipe.to(device)`:**

| | Estimate |
|---|---|
| **VRAM** | `S_total_q + A` (all components on GPU at once) |
| **RAM** | Minimal |
| **Speed** | Good — smaller model, may have dequantization overhead |
| **Quality** | Noticeable degradation possible, especially int4. Try int8 first. |

**Inference with `enable_model_cpu_offload()`:**

| | Estimate |
|---|---|
| **VRAM** | `S_max_q + A` (largest component on GPU at a time) |
| **RAM** | `S_total_q` (all components stored on CPU) |
| **Speed** | Moderate — component transfers between CPU/GPU |
| **Quality** | Depends on quantization level |

## Step 3: Pick the best strategy

Given `VRAM_available` and `RAM_available`, filter strategies by what fits, then rank by the user's preference.

### Algorithm

```
1. Measure S_total, S_max, S_block, S_leaf, S_total_lc, S_max_lc, A for the pipeline
2. For each strategy (offloading, casting, AND quantization), compute estimated VRAM and RAM
3. Filter out strategies where VRAM > VRAM_available or RAM > RAM_available
4. Present ALL viable strategies to the user grouped by approach (offloading/casting vs quantization)
5. Let the user pick based on their preference:
   - Quality:    pick the one with highest precision that fits
   - Speed:      pick the one with lowest transfer overhead
   - Memory:     pick the one with lowest VRAM usage
   - Balanced:   pick the lightest technique that fits comfortably (target ~80% VRAM)
```

### Quantization size estimates

Always compute these alongside offloading strategies — don't treat quantization as a last resort.
Pick the largest components worth quantizing (typically transformer + text_encoder if LLM-based):

```
S_component_int8 = S_component * 0.5
S_component_nf4  = S_component * 0.25

S_total_int8 = sum of quantized components (int8) + remaining components (bf16)
S_total_nf4  = sum of quantized components (nf4) + remaining components (bf16)
S_max_int8   = max single component after int8 quantization
S_max_nf4    = max single component after nf4 quantization
```

RAM requirement for quantization loading: `RAM >= S_largest_component_bf16` (full-precision weights
must fit during quantization). If this doesn't hold, quantization is not viable unless pre-quantized
checkpoints are available.

### Quick decision flowchart

Offloading / casting path:
```
VRAM >= S_total + A?
  → YES: No optimization needed (maybe attention backend for speed)
  → NO:
    VRAM >= S_total_lc + A? (layerwise casting, everything on GPU)
      → YES: Layerwise casting, pipe.to("cuda") without explicit dtype
      → NO:
        VRAM >= S_max + A? (model CPU offload, full precision)
          → YES: Model CPU offloading
                  - Want less VRAM? → add layerwise casting too
          → NO:
            VRAM >= S_max_lc + A? (layerwise casting + model CPU offload)
              → YES: Layerwise casting + model CPU offloading
              → NO: Need group offloading
                RAM >= 3 * S_total? (enough for pinned copies + overhead)
                  → YES: group offload leaf_level + stream (fast)
                  → NO:
                    RAM >= S_total?
                      → YES: group offload leaf_level + stream + low_cpu_mem_usage
                             or group offload block_level (no stream)
                      → NO: Quantization required to reduce model size, then retry
```

Quantization path (evaluate in parallel with the above, not as a fallback):
```
RAM >= S_largest_component_bf16? (must fit full-precision weights during quantization)
  → NO: Cannot quantize — need more RAM or pre-quantized checkpoints
  → YES: Compute quantized sizes for target components (typically transformer + text_encoder)
    nf4 quantization:
      VRAM >= S_total_nf4 + A?  → pipe.to("cuda"), fastest (no offloading overhead)
      VRAM >= S_max_nf4 + A?    → model CPU offload, moderate speed
    int8 quantization:
      VRAM >= S_total_int8 + A?  → pipe.to("cuda"), fastest
      VRAM >= S_max_int8 + A?    → model CPU offload, moderate speed

Show all viable quantization options alongside offloading options so the user can compare
quality/speed/memory tradeoffs across approaches.
```
