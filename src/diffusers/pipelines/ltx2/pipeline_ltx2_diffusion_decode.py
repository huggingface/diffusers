# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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

from ...models.autoencoders import AutoencoderKLLTX2Video, LTX2VideoDiffusionDecoderModel
from ...utils import logging
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LTX2VideoDecodeOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LTX2VideoDiffusionDecodePipeline(DiffusionPipeline):
    r"""
    Decode LTX-2 video latents with the diffusion decoder introduced in LTX-2.4.

    Unlike a convolutional decoder this one is itself a small diffusion model: it denoises pixels conditioned on a
    context volume built from the latents, so it needs a scheduler and a generator. Pair it with any LTX-2 pipeline run
    with `output_type="latent"`, passing `denormalize=False` since that path already applied the latent statistics.

    Args:
        diffusion_decoder ([`LTX2VideoDiffusionDecoderModel`]):
            The diffusion video decoder.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler driving the decoder's denoising steps.
        vae ([`AutoencoderKLLTX2Video`], *optional*):
            Only consulted for the latent statistics used to denormalize. When omitted the pipeline falls back to the
            LTX-2 defaults, so a decode-only workflow does not have to load a second autoencoder.
    """

    model_cpu_offload_seq = "diffusion_decoder"
    _optional_components = ["vae"]

    def __init__(
        self,
        diffusion_decoder: LTX2VideoDiffusionDecoderModel,
        scheduler,
        vae: AutoencoderKLLTX2Video = None,
    ):
        super().__init__()
        self.register_modules(diffusion_decoder=diffusion_decoder, scheduler=scheduler, vae=vae)
        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def _latent_stats(self, device: torch.device, dtype: torch.dtype):
        """Latent mean/std/scaling, from `vae` when it is loaded and from the LTX-2 defaults when it is not."""
        if self.vae is not None:
            return self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        return (
            self.diffusion_decoder.latents_mean.to(device=device, dtype=dtype),
            self.diffusion_decoder.latents_std.to(device=device, dtype=dtype),
            self.diffusion_decoder.config.scaling_factor,
        )

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample.LTX2LatentUpsamplePipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        denormalize: bool = True,
    ):
        r"""
        Args:
            latents (`torch.Tensor`):
                Latents of shape `(B, C, F, H, W)`. Note that an LTX-2 pipeline run with `output_type="latent"` returns
                latents that are *already* denormalized, so pass `denormalize=False` for those.
            generator (`torch.Generator`, *optional*):
                The decoder samples the noise it denoises, so pass a generator to make decoding reproducible.
            denormalize (`bool`, *optional*, defaults to `True`):
                Whether to apply the latent statistics before decoding. Set to `False` if the latents are already
                denormalized.

        Returns:
            [`~pipelines.ltx2.pipeline_output.LTX2VideoDecodeOutput`] or `tuple`
        """
        device = self._execution_device
        latents = latents.to(device)

        if denormalize:
            latents_mean, latents_std, scaling_factor = self._latent_stats(device, latents.dtype)
            latents = self._denormalize_latents(latents, latents_mean, latents_std, scaling_factor)

        latents = latents.to(self.diffusion_decoder.dtype)
        video = self.diffusion_decoder.decode(latents, generator=generator, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return LTX2VideoDecodeOutput(frames=video)
