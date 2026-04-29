# Memory Calculator

Use this guide to measure VRAM and RAM requirements for each optimization strategy, then recommend the best fit for the user's hardware.

## Step 1: Get model sizes with hf-mem

Use [hf-mem](https://github.com/alvarobartt/hf-mem) to get per-component sizes from the Hub without loading the model:

```bash
uvx hf-mem --model-id <model-id>
```

> **Windows note:** the default terminal encoding (cp1252) crashes on the output's box-drawing characters. Prefix with `PYTHONUTF8=1`:
> ```bash
> PYTHONUTF8=1 uvx hf-mem --model-id <model-id>
> ```

The output breaks down each pipeline component (transformer, text_encoder, vae, etc.) with param counts and sizes per dtype. Do not rely on the model name or card — pipelines often contain components that are not obvious (e.g. a "28B transformer" model may also carry a 24 GB text encoder and a 6 GB connectors module).

### Reading the output: checkpoint dtype vs. loaded dtype

`hf-mem` reports sizes at the **checkpoint's native dtype**. When loading with `torch_dtype=torch.bfloat16` (the standard), F32 components are halved. To get the actual loaded size:

- If a component shows **BF16** in the output → use as-is
- If a component shows **F32** in the output → divide by 2 for BF16 loaded size
- Quick formula: `total_params * 2 bytes` gives the full-BF16 pipeline size regardless of checkpoint dtype

Example (LTX-2): `hf-mem` reports 88.38 GiB total, but the text encoder is mostly F32 (45.4 GiB for 12.19B params). Loaded at BF16, the actual pipeline size is ~65 GiB.

### Deriving the values needed for strategy calculations

From the `hf-mem` output, record:

- **`S_total`** — sum of all component sizes at loaded dtype (use `total_params * 2 bytes` if loading at BF16)
- **`S_max`** — size of the largest single component at loaded dtype
- **`S_block`** — estimate as `S_transformer / num_blocks`; check the model config or card for `num_hidden_layers` / `num_transformer_layers`
- **`S_leaf`** — for most modern transformers this is very small (<0.05 GB); estimate as `S_block / 8` when needed for leaf_level offloading calculations
- **`S_total_lc`** — estimate for layerwise casting (fp8 storage, ~45% of BF16 size due to norm/embed layers being skipped): `S_total_lc ≈ S_total * 0.45`
- **`S_max_lc`** — same estimate applied to the largest component: `S_max_lc ≈ S_max * 0.45`
- **`A`** — activation memory (cannot be measured ahead of time — estimate conservatively):
  - **Video models**: scales with resolution × frames. A 5-second 960×544 video at 24fps uses ~7-8 GB. Higher resolution or longer duration = more.
  - **Image models**: scales with resolution. ~2-4 GB at 1024×1024, ~8-16 GB at 2048×2048.
  - **Edit/inpainting models**: budget extra for reference image(s) in activations.
  - When in doubt: `A ≈ 5-8 GB` for video, `A ≈ 2-4 GB` for images.

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

Reduces weight memory by casting to fp8. Norm and embedding layers are automatically skipped, so the reduction is less than 50% — use `S_component * 0.45` as the estimate.

**`pipe.to()` caveat:** `pipe.to(device)` internally calls `module.to(device, dtype)` where dtype is `None` when not explicitly passed. This preserves fp8 weights. However, if the user passes dtype explicitly (e.g., `pipe.to("cuda", torch.bfloat16)` or the pipeline has internal dtype overrides), the fp8 storage will be overridden back to bf16. When in doubt, combine with `enable_model_cpu_offload()` which safely moves one component at a time without dtype overrides.

**Case 1: Everything on GPU** (if `S_total_lc + A <= VRAM`)

| | Estimate |
|---|---|
| **VRAM** | `S_total_lc + A` (use `S_total * 0.45` estimate) |
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
1. Run hf-mem to get per-component sizes; derive S_total, S_max, S_block, S_leaf, S_total_lc, S_max_lc, A
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
