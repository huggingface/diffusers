# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union

import torch

from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLLTXVideo
from ...utils import deprecate, get_logger
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .modeling_latent_upsampler import LTXLatentUpsamplerModel
from .pipeline_output import LTXPipelineOutput


logger = get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LTXLatentUpsamplePipeline(DiffusionPipeline):
    model_cpu_offload_seq = ""

    def __init__(
        self,
        vae: AutoencoderKLLTXVideo,
        latent_upsampler: LTXLatentUpsamplerModel,
    ) -> None:
        super().__init__()

        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        video = video.to(device=device, dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        return init_latents

    def adain_filter_latent(self, latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):
        """
        Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
        tensor.

        Args:
            latent (`torch.Tensor`):
                Input latents to normalize
            reference_latents (`torch.Tensor`):
                The reference latents providing style statistics.
            factor (`float`):
                Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

        Returns:
            torch.Tensor: The transformed latent tensor
        """
        result = latents.clone()

        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)  # index by original dim order
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = torch.lerp(latents, result, factor)
        return result

    def tone_map_latents(self, latents: torch.Tensor, compression: float) -> torch.Tensor:
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range in a perceptually
        smooth way using a sigmoid-based compression.

        This is useful for regularizing high-variance latents or for conditioning outputs during generation, especially
        when controlling dynamic behavior with a `compression` factor.

        Args:
            latents : torch.Tensor
                Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
            compression : float
                Compression strength in the range [0, 1].
                - 0.0: No tone-mapping (identity transform)
                - 1.0: Full compression effect

        Returns:
            torch.Tensor
                The tone-mapped latent tensor of the same shape as input.
        """
        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

    def check_inputs(self, video, height, width, latents, tone_map_compression_ratio):
        if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` can be provided.")
        if video is None and latents is None:
            raise ValueError("One of `video` or `latents` has to be provided.")

        if not (0 <= tone_map_compression_ratio <= 1):
            raise ValueError("`tone_map_compression_ratio` must be in the range [0, 1]")

    @torch.no_grad()
    def __call__(
        self,
        video: Optional[List[PipelineImageInput]] = None,
        height: int = 512,
        width: int = 704,
        latents: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        adain_factor: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        self.check_inputs(
            video=video,
            height=height,
            width=width,
            latents=latents,
            tone_map_compression_ratio=tone_map_compression_ratio,
        )

        if video is not None:
            # Batched video input is not yet tested/supported. TODO: take a look later
            batch_size = 1
        else:
            batch_size = latents.shape[0]
        device = self._execution_device

        if video is not None:
            num_frames = len(video)
            if num_frames % self.vae_temporal_compression_ratio != 1:
                num_frames = (
                    num_frames // self.vae_temporal_compression_ratio * self.vae_temporal_compression_ratio + 1
                )
                video = video[:num_frames]
                logger.warning(
                    f"Video length expected to be of the form `k * {self.vae_temporal_compression_ratio} + 1` but is {len(video)}. Truncating to {num_frames} frames."
                )
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=torch.float32)

        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        latents = latents.to(self.latent_upsampler.dtype)
        latents_upsampled = self.latent_upsampler(latents)

        if adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
        else:
            latents = latents_upsampled

        if tone_map_compression_ratio > 0.0:
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)

        if output_type == "latent":
            latents = self._normalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
        else:
            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
