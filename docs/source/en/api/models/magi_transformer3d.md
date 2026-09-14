# MagiTransformer3DModel

MAGI-1 uses parallel video self-attention and text cross-attention, grouped-query attention, learned 3D rotary
embeddings, and chunk-level timestep conditioning. The 4.5B model has 34 blocks and GELU feed-forward layers; the
24B model has 48 blocks, SwiGLU, and duplicated latent input channels.

This implementation covers the non-quantized base and distilled architectures. It does not include MAGI's transport
scheduler, three-way guidance, distributed context/pipeline parallelism, or the original FP8 execution engine.

## Convert and load

Pass the official inference weight directory and its matching example configuration:

```bash
python scripts/convert_magi_to_diffusers.py transformer \
  --checkpoint_path /path/to/MAGI-1/ckpt/magi/4.5B_base/inference_weight \
  --config_path /path/to/MAGI-1/example/4.5B/4.5B_base_config.json \
  --output_path /path/to/magi-transformer-diffusers
```

The converter loads weights strictly and checks every tensor after saving and reloading. It retains the reference's
self/cross-attention output interleave. The text key/value projection is split into the eight linear calls used by the
reference; the other projection tensors are not reordered.

```python
import torch
from diffusers import MagiTransformer3DModel

transformer = MagiTransformer3DModel.from_pretrained(
    "/path/to/magi-transformer-diffusers", torch_dtype=torch.bfloat16
).to("cuda")
with torch.no_grad():
    prediction = transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_mask,
    ).sample
```

Latents use `(batch, 16, frames, height, width)` for both model sizes. The 24B channel duplication, output truncation,
and internal `x_rescale_factor` are handled by the model. The VAE-to-diffusion latent scaling still belongs outside it.
Embedding and output projections, rotary frequencies, self-attention Q/K normalization, and residual post-normalization
retain the reference's high-precision behavior. The attention output projection also computes in FP32 while retaining
its BF16 checkpoint weights, and timestep frequencies are rounded to the model dtype before their FP32 MLP. Load with `torch_dtype` instead of casting all weights with `.bfloat16()`.

The output `sample` is flow velocity, not a clean-latent prediction. Apply guidance to this velocity and pass it
directly to `MagiEulerScheduler.step`; do not divide a prediction residual by `1 - timestep`.

## Chunks and conditioning

`timestep` has shape `(batch,)` for one chunk or `(batch, chunks)` for equally sized temporal chunks. Values follow
the reference's [0, 1] convention; the model applies the factor of 1000 in its sinusoidal embedding. Text features can
be shared across chunks with shape `(batch, length, channels)` or supplied per chunk as `(batch, chunks, length, channels)`.
Boolean text masks have the corresponding shape without the channel dimension; `True` keeps a token.

Self-attention is chunk-causal by default: tokens attend to their entire current chunk and all preceding chunks.
`kv_ranges` overrides this with one exclusive `(start, end)` token range per current chunk. Ranges are shared across
batch items and index the concatenation of cached and current video tokens. The generation scheduler is responsible
for choosing the reference's timestep-dependent sliding windows.

`caption_dropout_mask` selects the learned conditional or unconditional *adaptive* embedding. As in official inference,
it does not replace text features in cross-attention. Pass the appropriate text features separately for guidance.

Distilled checkpoints also require `timestep_delta`. This is the extra timestep passed to the same embedding MLP,
not a difference between consecutive diffusion timesteps. In the official helper it is `num_steps / 2`, except when
`num_steps == 12`, where it is `8 / distill_interval`. These are the official distilled sampler's conventions.
The caller must supply this value; `MagiEulerScheduler` and the current base pipeline do not implement distilled sampling.

## Prefix cache

Use `use_cache=True` to return a tuple of `(key, value)` tensors, one pair per layer. Each tensor has shape
`(batch, tokens, key_value_heads, head_dim)`. Passing the tuple as `kv_cache` prepends those entries to current keys
and values and offsets temporal rotary positions accordingly. Input caches are not modified in place.

Only retain and reuse entries for clean, finalized prefix chunks. By default the output cache also includes current
chunks; `cache_token_count` can retain only a leading clean prefix and `cache_device="cpu"` can offload it per layer.
Caches computed with different conditioning are not interchangeable. The MAGI base pipeline intentionally shares one
null-caption clean-prefix cache between its two prefix-conditioned branches, matching the official sampler. Its
independent branch uses no cache. Do not reuse a cache across different spatial resolutions or batches.

The default attention backend works without MAGI-specific CUDA extensions. Explicitly selecting `flash` or
`flash_varlen` with `set_attention_backend` also uses FlashAttention's rotary kernel, matching the reference's fused
rounding. The native PyTorch rotary path can produce different low-precision outputs. Text padding masks must be honored;
backends that reject masks, such as `flash`, require inputs without padding and `encoder_attention_mask=None`.
Use a mask-capable backend when padding is present. Gradient checkpointing, standard device mapping, and group
offloading are supported; MAGI's distributed cache engine is not included.

## Compilation with text masks

Packing valid text tokens before projection preserves the reference GEMM shapes, but its output length depends on
the mask values. Enable Dynamo's dynamic-output-shape capture when compiling masked inputs with `fullgraph=True`:

```python
with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
    compiled_transformer = torch.compile(transformer, fullgraph=True)
    with torch.no_grad():
        prediction = compiled_transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            encoder_attention_mask=prompt_mask,
        ).sample
```

Keep this context active during execution, including calls that may trigger recompilation. The same requirement applies
to `compile_repeated_blocks(fullgraph=True)`. This is an explicit caller setting; importing MAGI does not change global
Dynamo configuration. Use the native attention backend for this path. Fused FlashAttention compilation is not validated.
Compilation can change floating-point rounding; eager FlashAttention remains the official numerical-parity path.

## MagiTransformer3DModel

[[autodoc]] MagiTransformer3DModel
    - all
