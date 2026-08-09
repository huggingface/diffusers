# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler, LTX2VideoDiffusionDecoderModel
from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import LTX2VideoDiffusionDecodePipeline

from ...testing_utils import enable_full_determinism, torch_device


enable_full_determinism()


DECODER_CONFIG = {
    "out_channels": 3,
    "latent_channels": 8,
    "patch_size": 2,
    "decoder_head_dim": 16,
    "decoder_stage_channels": (64, 32, 16, 16, 16),
    "decoder_stage_depths": (1, 1, 1, 1, 2),
    "decoder_stage_kernels": ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    "decoder_upsample_strides": ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
    "decoder_upsample_channel_reductions": (2, 2, 1, 1),
    "decoder_stage5_kernel": (3, 3, 3),
    "decoder_t_emb_dim": 32,
    "spatial_compression_ratio": 16,
    "temporal_compression_ratio": 8,
}


def _build(with_vae: bool = False):
    torch.manual_seed(0)
    decoder = LTX2VideoDiffusionDecoderModel(**DECODER_CONFIG).to(torch_device).eval()
    # Non-trivial statistics, so a run that skipped denormalization would not accidentally match.
    with torch.no_grad():
        decoder.latents_mean.copy_(torch.linspace(-0.1, 0.1, DECODER_CONFIG["latent_channels"]))
        decoder.latents_std.copy_(torch.linspace(0.5, 1.5, DECODER_CONFIG["latent_channels"]))

    vae = None
    if with_vae:
        torch.manual_seed(0)
        vae = (
            AutoencoderKLLTX2Video(
                in_channels=3,
                out_channels=3,
                latent_channels=DECODER_CONFIG["latent_channels"],
                block_out_channels=(8,),
                decoder_block_out_channels=(8,),
                layers_per_block=(1,),
                decoder_layers_per_block=(1, 1),
                spatio_temporal_scaling=(True,),
                decoder_spatio_temporal_scaling=(True,),
                decoder_inject_noise=(False, False),
                downsample_type=("spatial",),
                upsample_residual=(False,),
                upsample_factor=(1,),
                timestep_conditioning=False,
                patch_size=1,
                patch_size_t=1,
            )
            .to(torch_device)
            .eval()
        )

    return LTX2VideoDiffusionDecodePipeline(
        diffusion_decoder=decoder, scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae
    )


def _latents():
    return torch.randn(1, 8, 2, 3, 3, generator=torch.Generator().manual_seed(1)).to(torch_device)


def test_decode_without_vae():
    """`vae` is optional: the pipeline must fall back to the decoder's own latent statistics."""
    pipe = _build(with_vae=False)
    assert pipe.vae is None
    frames = pipe(_latents(), generator=torch.Generator(torch_device).manual_seed(0), output_type="np").frames[0]
    assert frames.shape == (9, 48, 48, 3)


def test_decode_with_vae_uses_its_statistics():
    """When a `vae` is supplied its statistics are used instead of the decoder's."""
    latents = _latents()
    without = _build(with_vae=False)(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    with_vae = _build(with_vae=True)(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt"
    ).frames
    # The dummy VAE's stats are mean 0 / std 1, the decoder's are not, so the two must disagree.
    assert not torch.equal(without, with_vae)


def test_decode_is_reproducible_with_a_generator():
    """The decoder samples the noise it denoises, so only a seeded generator makes it deterministic."""
    pipe, latents = _build(), _latents()
    first = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    same = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    other = pipe(latents, generator=torch.Generator(torch_device).manual_seed(7), output_type="pt").frames
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_denormalize_can_be_skipped():
    """`denormalize=False` must leave the latents alone for callers that already denormalized."""
    pipe, latents = _build(), _latents()
    normalized = pipe(latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt").frames
    raw = pipe(
        latents, generator=torch.Generator(torch_device).manual_seed(0), output_type="pt", denormalize=False
    ).frames
    assert not torch.equal(normalized, raw)
