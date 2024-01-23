# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import math
from typing import Any, Dict, Tuple, Union

import torch
import torch.fft as fft

from ...utils.torch_utils import randn_tensor


class FreeInitMixin:
    r"""
    Base class for FreeInit related utilities. A pipeline that derives from this class would mean that
    it supports the FreeInit mechanism as described in https://arxiv.org/abs/2312.07537.

    The methods exposed by a pipeline that derives from this class are:
        - `free_init_enabled`: Returns whether or not FreeInit has been enabled for generation.
        - `enable_free_init`: Enables the usage of the FreeInit mechanism.
        - `disable_free_init`: Disables the usage of the FreeInit mechanism.
    """

    @property
    def free_init_enabled(self):
        return hasattr(self, "_free_init_num_iters") and self._free_init_num_iters is not None

    def enable_free_init(
        self,
        num_iters: int = 3,
        use_fast_sampling: bool = False,
        method: str = "butterworth",
        order: int = 4,
        spatial_stop_frequency: float = 0.25,
        temporal_stop_frequency: float = 0.25,
    ):
        """Enables the FreeInit mechanism as in https://arxiv.org/abs/2312.07537.

        This implementation has been adapted from the [official repository](https://github.com/TianxingWu/FreeInit).

        Args:
            num_iters (`int`, *optional*, defaults to `3`):
                Number of FreeInit noise re-initialization iterations.
            use_fast_sampling (`bool`, *optional*, defaults to `False`):
                Whether or not to speedup sampling procedure at the cost of probably lower quality results. Enables
                the "Coarse-to-Fine Sampling" strategy, as mentioned in the paper, if set to `True`.
            method (`str`, *optional*, defaults to `butterworth`):
                Must be one of `butterworth`, `ideal` or `gaussian` to use as the filtering method for the
                FreeInit low pass filter.
            order (`int`, *optional*, defaults to `4`):
                Order of the filter used in `butterworth` method. Larger values lead to `ideal` method behaviour
                whereas lower values lead to `gaussian` method behaviour.
            spatial_stop_frequency (`float`, *optional*, defaults to `0.25`):
                Normalized stop frequency for spatial dimensions. Must be between 0 to 1. Referred to as `d_s` in
                the original implementation.
            temporal_stop_frequency (`float`, *optional*, defaults to `0.25`):
                Normalized stop frequency for temporal dimensions. Must be between 0 to 1. Referred to as `d_t` in
                the original implementation.
        """
        self._free_init_num_iters = num_iters
        self._free_init_use_fast_sampling = use_fast_sampling
        self._free_init_method = method
        self._free_init_order = order
        self._free_init_spatial_stop_frequency = spatial_stop_frequency
        self._free_init_temporal_stop_frequency = temporal_stop_frequency

    def disable_free_init(self):
        """Disables the FreeInit mechanism if enabled."""
        self._free_init_num_iters = None

    @staticmethod
    def _get_freeinit_freq_filter(
        shape: Tuple[int, ...],
        device: Union[str, torch.dtype],
        filter_type: str,
        order: float,
        spatial_stop_frequency: float,
        temporal_stop_frequency: float,
    ) -> torch.Tensor:
        r"""Returns the FreeInit filter based on filter type and other input conditions."""

        T, H, W = shape[-3], shape[-2], shape[-1]
        mask = torch.zeros(shape)

        if spatial_stop_frequency == 0 or temporal_stop_frequency == 0:
            return mask

        if filter_type == "butterworth":

            def retrieve_mask(x):
                return 1 / (1 + (x / spatial_stop_frequency**2) ** order)
        elif filter_type == "gaussian":

            def retrieve_mask(x):
                return math.exp(-1 / (2 * spatial_stop_frequency**2) * x)
        elif filter_type == "ideal":

            def retrieve_mask(x):
                return 1 if x <= spatial_stop_frequency * 2 else 0
        else:
            raise NotImplementedError("`filter_type` must be one of gaussian, butterworth or ideal")

        for t in range(T):
            for h in range(H):
                for w in range(W):
                    d_square = (
                        ((spatial_stop_frequency / temporal_stop_frequency) * (2 * t / T - 1)) ** 2
                        + (2 * h / H - 1) ** 2
                        + (2 * w / W - 1) ** 2
                    )
                    mask[..., t, h, w] = retrieve_mask(d_square)

        return mask.to(device)

    @staticmethod
    def _freq_mix_3d(x: torch.Tensor, noise: torch.Tensor, LPF: torch.Tensor) -> torch.Tensor:
        r"""Noise reinitialization."""
        # FFT
        x_freq = fft.fftn(x, dim=(-3, -2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
        noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
        noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

        # frequency mix
        HPF = 1 - LPF
        x_freq_low = x_freq * LPF
        noise_freq_high = noise_freq * HPF
        x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain

        # IFFT
        x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
        x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

        return x_mixed

    def _free_init_loop(
        self,
        height: int,
        width: int,
        num_frames: int,
        num_channels_latents: int,
        batch_size: int,
        num_videos_per_prompt: int,
        denoise_args: Dict[str, Any],
        device: Union[str, torch.dtype],
        generator: torch.Generator,
    ):
        """Denoising loop for AnimateDiff using FreeInit noise reinitialization technique."""

        latents = denoise_args.get("latents")
        prompt_embeds = denoise_args.get("prompt_embeds")
        timesteps = denoise_args.get("timesteps")
        num_inference_steps = denoise_args.get("num_inference_steps")
        H = height // self.vae_scale_factor
        W = width // self.vae_scale_factor
        bs = batch_size * num_videos_per_prompt

        latent_shape = (bs, num_channels_latents, num_frames, H, W)
        free_init_filter_shape = (1, num_channels_latents, num_frames, H, W)
        free_init_freq_filter = self._get_freeinit_freq_filter(
            shape=free_init_filter_shape,
            device=device,
            filter_type=self._free_init_method,
            order=self._free_init_order,
            spatial_stop_frequency=self._free_init_spatial_stop_frequency,
            temporal_stop_frequency=self._free_init_temporal_stop_frequency,
        )

        with self.progress_bar(total=self._free_init_num_iters) as free_init_progress_bar:
            for i in range(self._free_init_num_iters):
                # For the first FreeInit iteration, the original latent is used without modification.
                # Subsequent iterations apply the noise reinitialization technique.
                if i == 0:
                    initial_noise = latents.detach().clone()
                else:
                    current_diffuse_timestep = (
                        self.scheduler.config.num_train_timesteps - 1
                    )  # diffuse to t=999 noise level
                    diffuse_timesteps = torch.full((batch_size,), current_diffuse_timestep).long()
                    z_T = self.scheduler.add_noise(
                        original_samples=latents, noise=initial_noise, timesteps=diffuse_timesteps.to(device)
                    ).to(dtype=torch.float32)
                    z_rand = randn_tensor(latent_shape, generator, device, torch.float32)
                    latents = self._freq_mix_3d(z_T, z_rand, LPF=free_init_freq_filter)
                    latents = latents.to(prompt_embeds.dtype)

                # Coarse-to-Fine Sampling for faster inference (can lead to lower quality)
                if self._free_init_use_fast_sampling:
                    current_num_inference_steps = int(num_inference_steps / self._free_init_num_iters * (i + 1))
                    self.scheduler.set_timesteps(current_num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps
                    denoise_args.update({"timesteps": timesteps, "num_inference_steps": current_num_inference_steps})

                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                denoise_args.update({"latents": latents, "num_warmup_steps": num_warmup_steps})
                latents = self._denoise_loop(**denoise_args)

                free_init_progress_bar.update()

        return latents
