# LTX-2.4 diffusion VAE decoder — design decisions and evidence

Notes for the PR adding `AutoencoderKLLTX2VideoDiffusionDecoder`. Every decision below is recorded with the
reason, and where a number is quoted it comes from a run that is reproducible with the scripts named here.

## Context

LTX-2.4 replaced the video VAE's decoder: rc1's convolutional decoder became a neighborhood-attention
*diffusion* decoder (`NADiffusionDecoder` / `CausalDiffusionVAE`, `natten`-backed). The VAE **encoder is byte
identical** to rc1's — 27/27 sampled `vae.encoder.*` tensors hash-equal — so both decoders consume the same
latents and the latent space is unchanged. `diffusers` had no class for the new decoder, which is why
`LTX-2.4-RC2-Diffusers` shipped with rc1's conv `vae/` as a stopgap.

## Model design

**A separate model class, not a `decoder_type` flag on `AutoencoderKLLTX2Video`.**
`AutoencoderKLLTX2Video.__init__` already carries ~25 config entries, a dozen of them `decoder_*`. Adding a
second, disjoint set of decoder parameters behind a flag would make every config ambiguous about which half
applies. A separate class keeps each config self-describing and — the deciding factor — leaves the existing
conv path byte-for-byte untouched, so this PR cannot regress the in-flight LTX-2.4 integration PR. The
encoder is *reused*, not duplicated: the class imports `LTX2VideoEncoder3d`.

**Two attention processors, following the merged FLUX.3 video VAE.**
`autoencoder_kl_flux3_video.py` had already solved the same problem (3D neighborhood attention in a diffusers
video VAE), so this mirrors its structure rather than inventing one: `AttentionModuleMixin`,
`_default_processor_cls`, `_available_processors`, `_supports_qkv_fusion = False` (both processors read
`to_q`/`to_k`/`to_v` directly and have no fused path, so QKV fusion would build an unused `to_qkv` and
silently no-op). It also contributed `is_natten_available()`, added here to `utils/import_utils.py`.

* `LTX2VideoVaeNeighborhoodAttnProcessor` (**default**) — FlexAttention via `dispatch_attention_fn`. No extra
  dependency, so the class is importable and usable with a stock install.
* `LTX2VideoVaeNeighborhoodNattenProcessor` — `natten.na3d`, the reference's own kernel, and therefore the
  bit-exact path.

**The FlexAttention mask reproduces NATTEN's windowing, it does not approximate it.** At a grid boundary
NATTEN *shifts* its window inward and keeps it full size rather than truncating it, so the mask uses
`start = clamp(q - k // 2, 0, size - k)`. Measured agreement between the two processors on real modules:
**4.8e-07** (context) and **7.2e-07** (diffusion step) — float32 kernel noise, not a semantic difference.

**The query is pre-scaled and attention runs with `scale=1.0`.** The reference applies `q * head_dim**-0.5`
inside its QKV+RoPE step and then passes `scale=1.0` to `na3d`. Both processors here do the same, in the same
order (norm → scale → rotate), because moving the scale changes float rounding.

**RoPE is computed unchunked.** The reference splits the W axis into a fixed 4 slabs purely so Dynamo keeps
`W` dynamic; the rotation is elementwise per position, so slabbing is a memory/compile strategy with no
numerical effect. Inverse frequencies are built in **float64** and cast once, matching the reference's numpy
path bit for bit; rotation happens in fp32 on interleaved pairs and casts back to the input dtype.

**The MLP is plain eager SwiGLU.** The reference's `SwiGLU` runs a fused Triton kernel (`TILED`/`TRITON`
modes). Ours is bitwise identical to the same expression evaluated eagerly; the reference's kernel differs
from that eager expression by **5.9e-04**. So the residual end-to-end delta is the reference's kernel, and a
diffusers port that matched it would have to reproduce a Triton kernel's accumulation order — not worth it.

**Latent normalization belongs to the pipeline, not the decoder.** The reference un-normalizes inside
`forward_pre_diffusion`; diffusers does it in the pipeline via `latents_mean`/`latents_std`. This class
follows the diffusers convention so it is interchangeable with `AutoencoderKLLTX2Video`. The parity harness
neutralizes the reference's statistics (mean 0 / std 1) to compare like with like.

**Single-step `x0` is the primary path; the Euler loop is kept for multi-step.** LTX-2.4 ships
`model_output_type="x0"` with `default_num_inference_steps=1`, so decode is stages 1-4 plus one stage-5 step
whose prediction *is* the output. Both fields are real config in the checkpoint, and the loop is ~10 lines.

**Tiling is refused, not approximated.** `enable_tiling` raises. The decoder's neighborhood attention rejects
any tile smaller than its kernel — including a short remnant tile — so tile sizes cannot be chosen freely
(rc1's 256 px spatial tiles leave a 5-latent remnant and raise). Untiled decode is the honest default; a
faithful port of the reference's halo/tile schedule is ~400 lines and can come later.

**No einops.** `_patchify`/`_unpatchify` reproduce the reference's `b c (f p) (h q) (w r) -> b (c p r q) f h w`
with reshape/permute. Note the channel packing order is `(channel, width_offset, height_offset)`.

### Bugs found while building, and the invariants added for them

* **Frame count.** T latent frames decode to `(T - 1) * ratio + 1` pixel frames, not `T * ratio` — the
  temporal upsamples each drop a duplicate leading frame. Caught by a shape mismatch on the first smoke test.
* **Stage widths.** Each upsample divides channels by its reduction factor, so
  `stage_channels[i + 1] == stage_channels[i] // upsample_channel_reductions[i]`. An inconsistent pair used
  to fail deep inside the first block; now it raises in `__init__` with the expected value.
* **`use_slicing`.** `decode()` reads it but `__init__` never set it, so every real `decode()` raised
  `AttributeError`. The synthetic parity harness calls the decoder module directly and never went through
  `decode()`, which is why only the real-weight run caught it — an argument for landing the generated
  model-level tests before further real runs.

## Converter

`scripts/convert_ltx2_diffusion_vae_to_diffusers.py`. Rules taken from the reference's
`DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER`:

* strip `vae.decoder.`; `t_embedder.mlp.0/2` → `t_embedder.timestep_embedder.linear_1/2` (the reference's
  `t_embedder` *is* diffusers' `PixArtAlphaCombinedTimestepSizeEmbeddings`, saved under shorter names, so it
  is reused rather than reimplemented);
* split the fused `qkv.{weight,bias}` into `to_q`/`to_k`/`to_v`; rename `attn.proj` → `attn.to_out.0` and
  `q_norm`/`k_norm` → `norm_q`/`norm_k`;
* drop `coarse_*` preview heads; fold static AdaLN gates into the following Linear (`W ← g·W`) and drop them.
  **LTX-2.4's sft checkpoint carries no gates and no coarse params**, so those two rules no-op here; they are
  kept because gated (standalone distilled) checkpoints need them.
* `per_channel_statistics` `mean-of-means`/`std-of-means` become the `latents_mean`/`latents_std` buffers.

**Encoder weights *and* encoder config come from an already-converted VAE folder** (`--encoder-vae`), not from
class defaults. The encoder is the same module, so its widths must match whatever that checkpoint was
converted with — the real `block_out_channels` is `[256, 512, 1024, 1024]`, which is *not* the class default,
and assuming otherwise produced size mismatches.

Result on `ltx-2.4-22b-sft-rc2.safetensors`: 309 raw decoder keys → **405 tensors, 0 missing, 0 unexpected**,
1.47 GB bf16.

## Repo layout

The diffusion decoder ships as a **second subfolder beside `vae/`**, so `from_pretrained` keeps returning the
conv decoder by default and the new one is opt-in:

```python
vae = AutoencoderKLLTX2VideoDiffusionDecoder.from_pretrained(repo, subfolder="vae_diffusion")
pipe = LTX2Pipeline.from_pretrained(repo, vae=vae)
```

## Two tracks

The standard and modular pipelines need differently-shaped model repos (`model_index.json` vs
`modular_model_index.json`), so the work is carried on two branches with the same commits:

| branch | base | pipeline |
|---|---|---|
| `ltx-2-4-diffusion-vae-standard` | `2bc3c93`, the LTX-2.4 integration PR head | `LTX2Pipeline` |
| `ltx-2-4-diffusion-vae-modular` | `ltx-2-4-modular`, the modular WIP PR | modular |

## Testing

**Parity, per the repo's `parity-testing` skill** — both implementations in one process, seeded inputs, one
model at a time. `integrations/ltx2_diffusion_vae_parity.py`, float32 on CUDA (neighborhood attention has no
CPU kernel, so the skill's CPU preference cannot apply):

| check | max_diff | bitwise |
|---|---|---|
| attention (NATTEN) vs reference | 0.000e+00 | yes |
| our SwiGLU vs eager math | 0.000e+00 | yes |
| *reference's fused SwiGLU vs eager math* | *5.9e-04* | *no* |
| context, stages 1-4 | 5.4e-04 | no |
| diffusion step | 7.3e-04 | no |
| FlexAttention vs NATTEN processor | 4.8e-07 | no |

All inside the skill's `< 1e-3` component bar, with the residual attributed to a specific cause.

**Real weights, standard track.** One DiT run at 512×704×25, 20 steps, seed 42, then the *same* latent tensor
decoded twice — so the decoder is the only variable. Both produced `(1, 3, 25, 512, 704)`: conv mean 81.2 /
std 50.6, diffusion mean 81.3 / std 49.2. `latents.pt` is kept so the reference implementation can decode the
identical latents instead of relying on seed-matching across implementations.

**Still to do:** model-level tests via `utils/generate_model_tests.py` (what CI runs); the native reference
decode of those saved latents; the modular track, which needs a `modular_model_index.json` repo for rc2.

## Environment notes

* **Never import pip `natten` and the Hub `kernels` build in one process** — they share static CUTLASS state
  and whichever loads second returns NaNs. Parity between them must be cross-process.
* The Hub kernel story: `kernels-community` PR #1031 ported NATTEN v0.21.7 and targets `shi-labs/natten`, but
  its upload step is failing, so that repo is **empty**. `kernels-staging/natten` (branch `v0`) works and has
  18 variants — torch 2.11/2.12/2.13 only, so it cannot serve the native stack's torch 2.9. None of this
  blocks this PR: the FlexAttention default means there is **no hard natten dependency** at all.
* Running the pipeline needs `huggingface_hub >= 1.26` (`get_cached_repo_tree`); the native rc2 venv pins
  1.22 and fails at import.
