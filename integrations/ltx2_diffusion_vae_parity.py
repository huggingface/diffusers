"""Component parity: the native `DiffusionVideoDecoder` vs diffusers `LTX2VideoDiffusionDecoder3d`.

Runs against either the LTX-2.4 or LTX-2.5 reference package: 2.5 reworked the decoder's staging API and
the shims under "Reference-version shims" below absorb the difference.

TEMPORARY / for-visibility only. This whole `integrations/` directory is meant to be committed
transiently while the modular LTX-2 integration is under review, and removed before the final merge.
It is NOT a pytest test and is not wired into CI — run it manually, and note that it imports the native
`ltx_core` reference package, which is not a diffusers dependency.

Both implementations are built in this one process from the same random weights and fed the same inputs,
per the repo's parity-testing convention. Run it in an environment that has *both* the native `ltx_core`
package and `natten` installed, e.g. the reference source venv:

    PYTHONPATH=src python integrations/ltx2_diffusion_vae_parity.py

Notes on making the comparison apples-to-apples:

* **NATTEN, not FlexAttention.** The native decoder calls `natten.na3d`, so the diffusers side is switched to
  `LTX2VideoVaeNeighborhoodNattenProcessor`. That means CUDA and float32 rather than the CPU/float32 the
  parity skill prefers — neighborhood attention has no CPU kernel. The flex processor is compared against the
  NATTEN one separately (`--compare-processors`), which is the check that the portable path is equivalent.
* **Never import the Hub `kernels` build of natten in this process.** pip `natten` and the Hub kernel share
  static CUTLASS state and whichever loads second returns garbage.
* **Per-channel statistics are neutralised.** The native decoder un-normalises the latent inside
  `forward_pre_diffusion`; on the diffusers side that is the pipeline's job (`latents_mean`/`latents_std`), so
  the native buffers are set to mean 0 / std 1 to take them out of the comparison.
* **Noise is injected, not sampled.** `x_t` is built once and handed to both sides, so the comparison covers
  the decoder math rather than RNG call order.
"""

import argparse

import torch

from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoDiffusionDecoder3d,
    LTX2VideoVaeNeighborhoodAttention,
    LTX2VideoVaeNeighborhoodNattenProcessor,
    LTX2VideoVaeSwiGLU,
)


# Small enough to run quickly, but structurally the real thing: 4 deterministic stages with the production
# stride/reduction pattern, then a 2-block diffusion stage.
CONFIG = dict(
    in_channels=8,
    out_channels=3,
    patch_size=2,
    head_dim=16,
    stage_channels=(64, 32, 16, 16, 16),
    stage_depths=(1, 2, 1, 1, 2),
    stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    upsample_strides=((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
    upsample_channel_reductions=(2, 2, 1, 1),
    stage5_kernel=(3, 3, 3),
    t_emb_dim=32,
    temporal_compression_ratio=8,
)
LATENT_SHAPE = (1, 8, 3, 4, 4)


def build_native(device, dtype):
    from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder

    decoder = DiffusionVideoDecoder(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        patch_size=CONFIG["patch_size"],
        head_dim=CONFIG["head_dim"],
        stage_channels=CONFIG["stage_channels"],
        stage_depths=CONFIG["stage_depths"],
        # The native signature takes one kernel per stage (its last entry is unused — stage 5's kernel always
        # comes from `stage5_kernel`) plus (stride, reduction) pairs for the upsamples.
        stage_kernels=(*CONFIG["stage_kernels"], CONFIG["stage5_kernel"]),
        upsamples=tuple(zip(CONFIG["upsample_strides"], CONFIG["upsample_channel_reductions"])),
        stage5_kernel=CONFIG["stage5_kernel"],
        t_emb_dim=CONFIG["t_emb_dim"],
        default_num_inference_steps=1,
        timestep_scale_multiplier=1000.0,
        model_output_type="x0",
    )
    # `PerChannelStatistics` registers its buffers with `torch.empty`, so they must be written before use.
    decoder.per_channel_statistics.get_buffer("mean-of-means").zero_()
    decoder.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    decoder = decoder.to(device=device, dtype=dtype).eval()
    _install_combined_pathway(decoder)
    return decoder


# --- Reference-version shims -------------------------------------------------------------------------
# The reference decoder's staging API changed between LTX-2.4 and LTX-2.5. 2.5 added selectable decode
# pathways (`DiffVAEMode`), which split stages 1-4 apart so the chunked pathway can defer stage 4, and
# renamed the diff-step input builder. The three shims below keep this harness runnable against either
# release; each falls back to the older name when the newer one is absent.


def _install_combined_pathway(native):
    """Put 2.5's `diff_blocks` on the COMBINED pathway, eagerly.

    2.5 ships three decode presets and installs one by swapping `__class__` on every block, so an
    as-constructed decoder has no `forward_combined` at all. This port implements the COMBINED pathway
    (combined context, full-volume attention), so that is what the comparison has to install — with
    compilation off, since we want to compare math and not Inductor output. 2.4 had no pathway system
    and needs nothing.
    """
    try:
        from ltx_core.model.video_vae.transformer.apply import apply_diffvae_config
        from ltx_core.model.video_vae.transformer.config import DiffVAEBlockKind, DiffVAEConfig
    except ImportError:
        return native  # 2.4-era reference: single hard-wired pathway

    return apply_diffvae_config(
        native,
        DiffVAEConfig(
            block=DiffVAEBlockKind.COMBINED,
            w_chunks=1,
            natten_backend=None,
            compile_blocks=False,
            compile_det_stages=False,
        ),
    )


def native_context(native, latent):
    """Stages 1-4 of the reference decoder, as one call.

    2.4 did the trailing pad-and-crop inside `forward_pre_diffusion`. 2.5 moved the pad out to the caller
    (so the tiled path can pad once per volume rather than once per tile) and left the matching crop
    behind `forward_stage_4(pad_trailing=True)`, keyed off the same frame count. So the two halves have to
    be paired explicitly here; padding by `_natten_trailing_pad_latent_frames` is what the diffusers side
    does internally via `trailing_pad_latent_frames`. Skipping the pad while keeping the crop silently
    removes real frames — at this config the context comes back with 3 frames instead of 17.
    """
    if hasattr(native, "forward_pre_diffusion"):
        return native.forward_pre_diffusion(latent)

    from ltx_core.model.video_vae import diffusion_tiling

    padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
        latent, native._natten_trailing_pad_latent_frames
    )
    return native.forward_stage_4(native.forward_stages_1_to_3(padded))


def native_step_input(native, context, x_t):
    """`[context | conv_in_x_t(patchify(x_t))]`, the single buffer `forward_diff_step` consumes."""
    build = getattr(native, "_combined_for_diff_step", None) or native._context_and_x_for_diff_step
    return build(context, x_t)


def convert_state_dict(native_state_dict):
    """Map native decoder parameter names onto the diffusers module names.

    Only four things move: the attention QKV lives directly on the attention module rather than under a
    `qkv` submodule, the output projection becomes `to_out.0`, the Q/K norms are renamed, and the
    per-channel statistics are dropped (the pipeline owns latent normalisation on the diffusers side).
    """
    converted = {}
    for key, value in native_state_dict.items():
        if key.startswith("per_channel_statistics."):
            continue
        new_key = key
        new_key = new_key.replace(".attn.qkv.to_q.", ".attn.to_q.")
        new_key = new_key.replace(".attn.qkv.to_k.", ".attn.to_k.")
        new_key = new_key.replace(".attn.qkv.to_v.", ".attn.to_v.")
        new_key = new_key.replace(".attn.proj.", ".attn.to_out.0.")
        new_key = new_key.replace(".attn.q_norm.", ".attn.norm_q.")
        new_key = new_key.replace(".attn.k_norm.", ".attn.norm_k.")
        converted[new_key] = value
    return converted


def build_diffusers(native_decoder, device, dtype, processor="natten"):
    decoder = LTX2VideoDiffusionDecoder3d(**CONFIG)
    missing, unexpected = decoder.load_state_dict(convert_state_dict(native_decoder.state_dict()), strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    decoder = decoder.to(device=device, dtype=dtype).eval()
    if processor == "natten":
        for module in decoder.modules():
            if isinstance(module, LTX2VideoVaeNeighborhoodAttention):
                module.set_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
    return decoder


def report(name, reference, candidate, tolerance=1e-3):
    max_diff = (reference.float() - candidate.float()).abs().max().item()
    bitwise = torch.equal(reference, candidate)
    status = "PASS" if max_diff < tolerance else "FAIL"
    print(f"  [{status}] {name:24s} max_diff={max_diff:.3e}  bitwise={bitwise}  shape={tuple(candidate.shape)}")
    return max_diff < tolerance


@torch.inference_mode()
def compare_components(device, dtype):
    """Per-module parity, which is what attributes any end-to-end delta to a specific cause.

    Attention comes out bitwise identical. The MLP does not, and the reason is on the reference side: its
    `SwiGLU` runs a fused Triton kernel whose result differs from the same expression evaluated eagerly.
    This checks our MLP against that eager expression, so a non-zero `native fused SwiGLU` row is expected
    and is the reference's kernel, not a porting error.
    """
    import torch.nn.functional as F
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
    from ltx_core.model.video_vae.transformer.swiglu import SwiGLU

    print("component parity:")
    hidden_states = torch.randn(1, 5, 8, 8, 32, device=device, dtype=dtype)

    native_attn = NeighborhoodAttention3D(32, (3, 3, 3), head_dim=16).to(device=device, dtype=dtype).eval()
    attn = LTX2VideoVaeNeighborhoodAttention(32, (3, 3, 3), head_dim=16).to(device=device, dtype=dtype).eval()
    # `convert_state_dict` matches on a leading `.attn.`, so give this standalone module a dummy parent and
    # strip it back off afterwards.
    renamed = convert_state_dict({f"block.attn.{key}": value for key, value in native_attn.state_dict().items()})
    attn.load_state_dict({key.removeprefix("block.attn."): value for key, value in renamed.items()})
    attn.set_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
    ok = report("attention (NATTEN)", native_attn(hidden_states), attn(hidden_states))

    native_mlp = SwiGLU(32, 64).to(device=device, dtype=dtype).eval()
    mlp = LTX2VideoVaeSwiGLU(32, 64).to(device=device, dtype=dtype).eval()
    mlp.load_state_dict(native_mlp.state_dict())
    eager = F.linear(
        F.silu(F.linear(hidden_states, native_mlp.w_gate.weight)) * F.linear(hidden_states, native_mlp.w_up.weight),
        native_mlp.w_down.weight,
    )
    ok &= report("SwiGLU vs eager math", eager, mlp(hidden_states))
    report("native fused SwiGLU", eager, native_mlp(hidden_states), tolerance=float("inf"))
    return ok


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--compare-processors",
        action="store_true",
        help="Also compare the diffusers FlexAttention processor against the NATTEN one.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype}")

    torch.manual_seed(0)
    ok_components = compare_components(device, dtype)

    generator = torch.Generator().manual_seed(42)
    latent = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)

    print("end-to-end parity:")
    torch.manual_seed(0)
    native = build_native(device, dtype)
    diffusers_decoder = build_diffusers(native, device, dtype, processor="natten")

    # Stage 1-4 context.
    native_ctx = native_context(native, latent).clone()
    diffusers_context = diffusers_decoder.forward_pre_diffusion(latent)
    ok = ok_components & report("pre_diffusion context", native_ctx, diffusers_context)

    # One stage-5 step on identical injected noise.
    num_frames = (LATENT_SHAPE[2] - 1) * CONFIG["temporal_compression_ratio"] + 1
    spatial = 8 * CONFIG["patch_size"]
    x_t = torch.randn(
        (LATENT_SHAPE[0], CONFIG["out_channels"], num_frames, LATENT_SHAPE[3] * spatial, LATENT_SHAPE[4] * spatial),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    timestep = torch.ones((LATENT_SHAPE[0],), device=device, dtype=torch.float32)

    native_pred = native.forward_diff_step(native_step_input(native, native_ctx, x_t), timestep).clone()
    diffusers_pred = diffusers_decoder.forward_diffusion_step(diffusers_context, x_t, timestep)
    ok &= report("diffusion step (x0)", native_pred, diffusers_pred)

    if args.compare_processors:
        flex_decoder = build_diffusers(native, device, dtype, processor="flex")
        flex_context = flex_decoder.forward_pre_diffusion(latent)
        ok &= report("flex vs natten context", diffusers_context, flex_context)
        flex_pred = flex_decoder.forward_diffusion_step(flex_context, x_t, timestep)
        ok &= report("flex vs natten step", diffusers_pred, flex_pred)

    print("PARITY OK" if ok else "PARITY FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
