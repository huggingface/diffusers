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

`scripts/convert_ltx2_to_diffusers.py --diffusion_vae`, alongside the existing `--vae`, rather than a second
script: it is the same checkpoint, and the encoder half must go through the same rules as the conv VAE. The
decoder rules are taken from the reference's `DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER`:

* `t_embedder.mlp.0/2` → `t_embedder.timestep_embedder.linear_1/2` (the reference's `t_embedder` *is*
  diffusers' `PixArtAlphaCombinedTimestepSizeEmbeddings`, saved under shorter names, so it is reused rather
  than reimplemented);
* split the fused `qkv.{weight,bias}` into `to_q`/`to_k`/`to_v`; rename `attn.proj` → `attn.to_out.0` and
  `q_norm`/`k_norm` → `norm_q`/`norm_k`;
* drop `coarse_*` preview heads; fold static AdaLN gates into the following Linear (`W ← g·W`) and drop them.
  **LTX-2.4's sft checkpoint carries no gates and no coarse params**, so those two rules no-op here; they are
  kept because gated (standalone distilled) checkpoints need them.

**The `decoder.` half and everything else are split before renaming.** The other half — the encoder and
`per_channel_statistics` (which becomes the `latents_mean`/`latents_std` buffers) — reuses
`LTX_2_3_VIDEO_VAE_RENAME_DICT` unchanged. None of the diffusion decoder's parameter names collide with that
dict's keys today, but splitting first makes that a property of the code rather than a coincidence.

**The encoder config is pinned explicitly in `get_ltx2_diffusion_video_vae_config`, not defaulted.** The real
`block_out_channels` is `(256, 512, 1024, 1024)` and `layers_per_block` is `(4, 6, 4, 2, 2)`, neither of which
is the class default (those are 2.0's), and assuming otherwise produced size mismatches. Those entries have to
stay in sync with `get_ltx2_video_vae_config("2.4")` — same encoder, same widths.

Result on `ltx-2.4-22b-sft-rc2.safetensors`: 309 raw decoder keys + 84 encoder keys + 2 statistics →
**491 tensors, 0 missing, 0 unexpected**, 1.47 GB bf16. Verified **bitwise identical** to what the earlier
standalone script produced (491/491 tensors equal, identical `config.json`), which is what licensed deleting
it.

## Pipeline integration

The pipeline's decode step needed a branch; this was **not** optional, and it was found by asking why the
documented usage below had never actually been run. Every earlier real-weight run called `vae.decode()`
directly, so it never went through `LTX2Pipeline`, and through the pipeline it did not work at all:

* `if not self.vae.config.timestep_conditioning:` — the diffusion decoder has no reason to carry that entry
  (it does its own noising), so this raised `AttributeError: 'FrozenDict' object has no attribute
  'timestep_conditioning'` before decode was ever reached;
* `self.vae.decode(latents, timestep, ...)` passes the timestep **positionally**, which on this class lands on
  `generator`.

The fix is in `pipeline_ltx2.py` (and the modular `decoders.py` on the modular branch): one
`isinstance(self.vae, AutoencoderKLLTX2VideoDiffusionDecoder)` flag that skips the `decode_timestep`
pre-noising and passes `generator=generator` instead of a timestep. **The conv path's own statements are left
untouched** — the flag only short-circuits the condition and picks the `decode` call — so it cannot regress
the in-flight LTX-2.4 PR.

Worth noting where that decode block comes from: `git log -L` puts it in **`c10bdd9b7` "Add LTX 2.0 Video
Pipelines" (#12915), already on `main`** — it is LTX-2.0 code, not something the LTX-2.4 integration PR wrote.
That PR does edit `pipeline_ltx2.py` heavily, but its last hunk there ends ~290 lines above this block, so the
two do not overlap.

Passing the generator is what makes decoding reproducible: two full pipeline runs at 320×448×17 from the same
seed now agree **bitwise** (`run_pipeline_diffusion_vae.py`), where a dropped generator would reseed the
decoder's noise on every call. Covered by `test_inference_with_diffusion_decoder_vae` in
`tests/pipelines/ltx2/test_ltx2.py`, which fails with the `AttributeError` above if the pipeline branch is
reverted.

**Real-resolution decode wants the NATTEN processor.** diffusers deliberately does not compile
`flex_attention` (it leaves that to the user), and uncompiled it materializes the full score matrix — at
stage 5's sequence length (17×80×112 ≈ 152k tokens at 320×448) that is not viable. So the FlexAttention
default keeps the class importable and correct with no new dependency, which is what it is for; full-resolution
decoding wants `natten` installed (or a compiled decoder). Both real-weight runs set the NATTEN processor
explicitly for this reason.

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

**Real weights, against the reference decoder.** `native_decode_latents.py` builds the reference
`DiffusionVideoDecoder` from the rc2 checkpoint and decodes the *same* saved latents, in one process, so it is
decoder-vs-decoder on production weights with nothing else varying (bf16, 512×704×25):

| metric | value |
|---|---|
| max abs diff | 1.43e-02 |
| mean abs diff | **5.60e-04** |
| cosine similarity | 0.999997 |
| native mean / std | -0.3527 / 0.3858 |
| ours mean / std | -0.3527 / 0.3859 |

The mean absolute difference matches the fused-SwiGLU kernel delta measured on random weights (5.9e-04), i.e.
the residual at full scale has the same, understood cause. Comfortably inside the parity skill's bf16
guidance (max_diff < 1e-1, cosine > 0.9999). Note the input contract differs by design: the reference
un-normalizes the latent itself, so it takes `latents.pt` as saved, while this class expects denormalized
input.

**Model-level tests, what CI runs.**
`tests/models/autoencoders/test_models_autoencoder_kl_ltx2_diffusion_decoder.py`, generated with
`utils/generate_model_tests.py` and then filled in: `ModelTesterMixin`, `MemoryTesterMixin`,
`AttentionTesterMixin`, and `NewAutoencoderTesterMixin` for slicing. **35 passed, 7 skipped.** Three things
had to be decided rather than filled in:

* **The class needed a `forward()`.** The mixins all call `model(**inputs)`, and only `encode`/`decode`
  existed. It is encode → `mode()` (or `sample()`) → decode, mirroring `AutoencoderKLLTX2Video.forward` minus
  the parameters that do not apply (`temb`, `decoder_causal`).
* **`forward` takes a `generator`, and the tests pass a seeded one.** This decoder denoises, so it draws
  noise: without a generator no two forward passes agree and every comparison in the suite is vacuous. This
  is the existing convention for stochastic VAEs (`AutoencoderKLConsistencyDecoder`,
  `NewAutoencoderTesterMixin`'s `_accepts_generator`), not a new one.
* **`MemoryTesterMixin.test_group_offloading` had to be fixed, in `tests/models/testing_utils/memory.py`.**
  It builds one `inputs_dict` and runs four forwards from it, so a stateful generator advanced between them
  and the outputs could not match. `test_group_offloading_with_disk` in the same file already re-seeded for
  exactly this reason, via a local helper; that helper is now module-level (`reseed_generator_input`) and used
  by both. It also had to read the signature off the *class*, not the instance: `enable_group_offload`
  replaces `model.forward` with a `(*args, **kwargs)` wrapper, so the instance lookup finds no `generator` on
  precisely the models these tests offload. With it, all four offload modes are **bitwise identical** to the
  un-offloaded output.

The smallest dummy config the decoder admits is set by its own invariants: neighborhood attention needs every
stage to be at least its kernel size in T/H/W, so with a kernel of 3 the smallest usable latent is 2×3×3 —
a 9×48×48 video at 16× spatial / 8× temporal compression. The suite runs in ~20 s.

**Modular track, real weights.** `run_modular_diffusion_vae.py`. No `modular_model_index.json` repo was
needed after all: `integrations/ltx2_t2v_parity.py` already builds the modular pipeline as
`LTX2Blocks().init_pipeline()` plus `update_components(...)` fed from a standard `LTX2Pipeline.from_pretrained`,
so the same route injects the diffusion decoder. Publishing a modular repo is about *distribution*, not about
being able to run this. Two runs at 320×448×17 from one seed: `(17, 3, 320, 448)`, mean 0.4902 / std 0.3105,
**max abs diff 0.000e+00** — the modular decode block passes the generator through as well.

There are no `tests/modular_pipelines/ltx2/` tests on the modular branch yet (the modular PR has not added the
harness or its tiny repo), so the modular decode branch has runtime verification but no CI coverage; the
standard branch's `test_inference_with_diffusion_decoder_vae` covers the equivalent logic.

## Environment notes

* **Never import pip `natten` and the Hub `kernels` build in one process** — they share static CUTLASS state
  and whichever loads second returns NaNs. Parity between them must be cross-process.
* The Hub kernel story: `kernels-community` PR #1031 ported NATTEN v0.21.7 and targets `shi-labs/natten`, but
  its upload step is failing, so that repo is **empty**. `kernels-staging/natten` (branch `v0`) works and has
  18 variants — torch 2.11/2.12/2.13 only, so it cannot serve the native stack's torch 2.9. None of this
  blocks this PR: the FlexAttention default means there is **no hard natten dependency** at all.
* Running the pipeline needs `huggingface_hub >= 1.26` (`get_cached_repo_tree`); the native rc2 venv pins
  1.22 and fails at import.
