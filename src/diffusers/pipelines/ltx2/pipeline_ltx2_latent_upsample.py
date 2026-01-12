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
from ...models import AutoencoderKLLTX2Video
from ...utils import get_logger, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..ltx.pipeline_output import LTXPipelineOutput
from ..pipeline_utils import DiffusionPipeline
from .latent_upsampler import LTX2LatentUpsamplerModel


logger = get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
        >>> from diffusers.pipelines.ltx2.export_utils import encode_video
        >>> from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        >>> from diffusers.utils import load_image

        >>> pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
        ... )
        >>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background."
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> frame_rate = 24.0
        >>> video, audio = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=768,
        ...     height=512,
        ...     num_frames=121,
        ...     frame_rate=frame_rate,
        ...     num_inference_steps=40,
        ...     guidance_scale=4.0,
        ...     output_type="pil",
        ...     return_dict=False,
        ... )

        >>> latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        ...     "Lightricks/LTX-2", subfolder="latent_upsampler", torch_dtype=torch.bfloat16
        ... )
        >>> upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        >>> upsample_pipe.vae.enable_tiling()
        >>> upsample_pipe.to(device="cuda", dtype=torch.bfloat16)

        >>> video = upsample_pipe(
        ...     video=video,
        ...     width=768,
        ...     height=512,
        ...     output_type="np",
        ...     return_dict=False,
        ... )[0]
        >>> video = (video * 255).round().astype("uint8")
        >>> video = torch.from_numpy(video)

        >>> encode_video(
        ...     video[0],
        ...     fps=frame_rate,
        ...     audio=audio[0].float().cpu(),
        ...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
        ...     output_path="video.mp4",
        ... )
        ```
"""


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


class LTX2LatentUpsamplePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "vae->latent_upsampler"

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        latent_upsampler: LTX2LatentUpsamplerModel,
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
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                # Convert token seq [B, S, D] to latent video [B, C, F, H, W]
                latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
                latent_height = height // self.vae_spatial_compression_ratio
                latent_width = width // self.vae_spatial_compression_ratio
                latents = self._unpack_latents(
                    latents, latent_num_frames, latent_height, latent_width, spatial_patch_size, temporal_patch_size
                )
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
        # NOTE: latent upsampler operates on the unnormalized latents, so don't normalize here
        # init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
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
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2_image2video.LTX2ImageToVideoPipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: Optional[List[PipelineImageInput]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        latents: Optional[torch.Tensor] = None,
        latents_normalized: bool = False,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        adain_factor: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            video (`List[PipelineImageInput]`, *optional*)
                The video to be upsampled (such as a LTX 2.0 first stage output). If not supplied, `latents` should be
                supplied.
            height (`int`, *optional*, defaults to `512`):
                The height in pixels of the input video (not the generated video, which will have a larger resolution).
            width (`int`, *optional*, defaults to `768`):
                The width in pixels of the input video (not the generated video, which will have a larger resolution).
            num_frames (`int`, *optional*, defaults to `121`):
                The number of frames in the input video.
            spatial_patch_size (`int`, *optional*, defaults to `1`):
                The spatial patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.
            temporal_patch_size (`int`, *optional*, defaults to `1`):
                The temporal patch size of the video latents. Used when `latents` is supplied if unpacking is
                necessary.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video latents. This can be supplied in place of the `video` argument. Can either be a
                patch sequence of shape `(batch_size, seq_len, hidden_dim)` or a video latent of shape `(batch_size,
                latent_channels, latent_frames, latent_height, latent_width)`.
            latents_normalized (`bool`, *optional*, defaults to `False`)
                If `latents` are supplied, whether the `latents` are normalized using the VAE latent mean and std. If
                `True`, the `latents` will be denormalized before being supplied to the latent upsampler.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            adain_factor (`float`, *optional*, defaults to `0.0`):
                Adaptive Instance Normalization (AdaIN) blending factor between the upsampled and original latents.
                Should be in [-10.0, 10.0]; supplying 0.0 (the default) means that AdaIN is not performed.
            tone_map_compression_ratio (`float`, *optional*, defaults to `0.0`):
                The compression strength for tone mapping, which will reduce the dynamic range of the latent values.
                This is useful for regularizing high-variance latents or for conditioning outputs during generation.
                Should be in [0, 1], where 0.0 (the default) means tone mapping is not applied and 1.0 corresponds to
                the full compression effect.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the upsampled video.
        """

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

        latents_supplied = latents is not None
        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        if latents_supplied and latents_normalized:
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
