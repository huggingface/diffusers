# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Callable, Dict, Optional, Union

import torch

from ..models.attention import BasicTransformerBlock, FreeNoiseTransformerBlock
from ..models.unets.unet_motion_model import (
    CrossAttnDownBlockMotion,
    DownBlockMotion,
    UpBlockMotion,
)
from ..pipelines.pipeline_utils import DiffusionPipeline
from ..utils import logging
from ..utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AnimateDiffFreeNoiseMixin:
    r"""Mixin class for [FreeNoise](https://arxiv.org/abs/2310.15169)."""

    def _enable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion]):
        r"""Helper function to enable FreeNoise in transformer blocks."""

        for motion_module in block.motion_modules:
            num_transformer_blocks = len(motion_module.transformer_blocks)

            for i in range(num_transformer_blocks):
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    motion_module.transformer_blocks[i].set_free_noise_properties(
                        self._free_noise_context_length,
                        self._free_noise_context_stride,
                        self._free_noise_weighting_scheme,
                    )
                else:
                    assert isinstance(motion_module.transformer_blocks[i], BasicTransformerBlock)
                    basic_transfomer_block = motion_module.transformer_blocks[i]

                    motion_module.transformer_blocks[i] = FreeNoiseTransformerBlock(
                        dim=basic_transfomer_block.dim,
                        num_attention_heads=basic_transfomer_block.num_attention_heads,
                        attention_head_dim=basic_transfomer_block.attention_head_dim,
                        dropout=basic_transfomer_block.dropout,
                        cross_attention_dim=basic_transfomer_block.cross_attention_dim,
                        activation_fn=basic_transfomer_block.activation_fn,
                        attention_bias=basic_transfomer_block.attention_bias,
                        only_cross_attention=basic_transfomer_block.only_cross_attention,
                        double_self_attention=basic_transfomer_block.double_self_attention,
                        positional_embeddings=basic_transfomer_block.positional_embeddings,
                        num_positional_embeddings=basic_transfomer_block.num_positional_embeddings,
                        context_length=self._free_noise_context_length,
                        context_stride=self._free_noise_context_stride,
                        weighting_scheme=self._free_noise_weighting_scheme,
                    ).to(device=self.device, dtype=self.dtype)

                    motion_module.transformer_blocks[i].load_state_dict(
                        basic_transfomer_block.state_dict(), strict=True
                    )

    def _disable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion]):
        r"""Helper function to disable FreeNoise in transformer blocks."""

        for motion_module in block.motion_modules:
            num_transformer_blocks = len(motion_module.transformer_blocks)

            for i in range(num_transformer_blocks):
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    free_noise_transfomer_block = motion_module.transformer_blocks[i]

                    motion_module.transformer_blocks[i] = BasicTransformerBlock(
                        dim=free_noise_transfomer_block.dim,
                        num_attention_heads=free_noise_transfomer_block.num_attention_heads,
                        attention_head_dim=free_noise_transfomer_block.attention_head_dim,
                        dropout=free_noise_transfomer_block.dropout,
                        cross_attention_dim=free_noise_transfomer_block.cross_attention_dim,
                        activation_fn=free_noise_transfomer_block.activation_fn,
                        attention_bias=free_noise_transfomer_block.attention_bias,
                        only_cross_attention=free_noise_transfomer_block.only_cross_attention,
                        double_self_attention=free_noise_transfomer_block.double_self_attention,
                        positional_embeddings=free_noise_transfomer_block.positional_embeddings,
                        num_positional_embeddings=free_noise_transfomer_block.num_positional_embeddings,
                    ).to(device=self.device, dtype=self.dtype)

                    motion_module.transformer_blocks[i].load_state_dict(
                        free_noise_transfomer_block.state_dict(), strict=True
                    )

    def _check_inputs_free_noise(
        self,
        prompt,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        num_frames,
    ) -> None:
        if not isinstance(prompt, (str, dict)):
            raise ValueError(f"Expected `prompt` to have type `str` or `dict` but found {type(prompt)=}")

        if negative_prompt is not None:
            if not isinstance(negative_prompt, (str, dict)):
                raise ValueError(
                    f"Expected `negative_prompt` to have type `str` or `dict` but found {type(negative_prompt)=}"
                )

        if prompt_embeds is not None or negative_prompt_embeds is not None:
            raise ValueError("`prompt_embeds` and `negative_prompt_embeds` is not supported in FreeNoise yet.")

        frame_indices = [isinstance(x, int) for x in prompt.keys()]
        frame_prompts = [isinstance(x, str) for x in prompt.values()]
        min_frame = min(list(prompt.keys()))
        max_frame = max(list(prompt.keys()))

        if not all(frame_indices):
            raise ValueError("Expected integer keys in `prompt` dict for FreeNoise.")
        if not all(frame_prompts):
            raise ValueError("Expected str values in `prompt` dict for FreeNoise.")
        if min_frame != 0:
            raise ValueError("The minimum frame index in `prompt` dict must be 0 as a starting prompt is necessary.")
        if max_frame >= num_frames:
            raise ValueError(
                f"The maximum frame index in `prompt` dict must be lesser than {num_frames=} and follow 0-based indexing."
            )

    def _encode_prompt_free_noise(
        self,
        prompt: Union[str, Dict[int, str]],
        num_frames: int,
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, Dict[int, str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ) -> torch.Tensor:
        if negative_prompt is None:
            negative_prompt = ""

        # Ensure that we have a dictionary of prompts
        if isinstance(prompt, str):
            prompt = {0: prompt}
        if isinstance(negative_prompt, str):
            negative_prompt = {0: negative_prompt}

        self._check_inputs_free_noise(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds, num_frames)

        # Sort the prompts based on frame indices
        prompt = dict(sorted(prompt.items()))
        negative_prompt = dict(sorted(negative_prompt.items()))

        # Ensure that we have a prompt for the last frame index
        prompt[num_frames - 1] = prompt[list(prompt.keys())[-1]]
        negative_prompt[num_frames - 1] = negative_prompt[list(negative_prompt.keys())[-1]]

        frame_indices = list(prompt.keys())
        frame_prompts = list(prompt.values())
        frame_negative_indices = list(negative_prompt.keys())
        frame_negative_prompts = list(negative_prompt.values())

        # Generate and interpolate positive prompts
        prompt_embeds, _ = self.encode_prompt(
            prompt=frame_prompts,
            device=device,
            num_images_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        shape = (num_frames, *prompt_embeds.shape[1:])
        prompt_interpolation_embeds = prompt_embeds.new_zeros(shape)

        for i in range(len(frame_indices) - 1):
            start_frame = frame_indices[i]
            end_frame = frame_indices[i + 1]
            start_tensor = prompt_embeds[i].unsqueeze(0)
            end_tensor = prompt_embeds[i + 1].unsqueeze(0)

            prompt_interpolation_embeds[start_frame : end_frame + 1] = self._free_noise_prompt_interpolation_callback(
                start_frame, end_frame, start_tensor, end_tensor
            )

        # Generate and interpolate negative prompts
        negative_prompt_embeds = None
        negative_prompt_interpolation_embeds = None

        if do_classifier_free_guidance:
            _, negative_prompt_embeds = self.encode_prompt(
                prompt=[""] * len(frame_negative_prompts),
                device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=frame_negative_prompts,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )

            negative_prompt_interpolation_embeds = negative_prompt_embeds.new_zeros(shape)

            for i in range(len(frame_negative_indices) - 1):
                start_frame = frame_negative_indices[i]
                end_frame = frame_negative_indices[i + 1]
                start_tensor = negative_prompt_embeds[i].unsqueeze(0)
                end_tensor = negative_prompt_embeds[i + 1].unsqueeze(0)

                negative_prompt_interpolation_embeds[
                    start_frame : end_frame + 1
                ] = self._free_noise_prompt_interpolation_callback(start_frame, end_frame, start_tensor, end_tensor)

        prompt_embeds = prompt_interpolation_embeds
        negative_prompt_embeds = negative_prompt_interpolation_embeds

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds, negative_prompt_embeds

    def _prepare_latents_free_noise(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        context_num_frames = (
            self._free_noise_context_length if self._free_noise_context_length == "repeat_context" else num_frames
        )

        shape = (
            batch_size,
            num_channels_latents,
            context_num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if self._free_noise_noise_type == "random":
                return latents
        else:
            if latents.size(2) == num_frames:
                return latents
            elif latents.size(2) != self._free_noise_context_length:
                raise ValueError(
                    f"You have passed `latents` as a parameter to FreeNoise. The expected number of frames is either {num_frames} or {self._free_noise_context_length}, but found {latents.size(2)}"
                )
            latents = latents.to(device)

        if self._free_noise_noise_type == "shuffle_context":
            for i in range(self._free_noise_context_length, num_frames, self._free_noise_context_stride):
                # ensure window is within bounds
                window_start = max(0, i - self._free_noise_context_length)
                window_end = min(num_frames, window_start + self._free_noise_context_stride)
                window_length = window_end - window_start

                if window_length == 0:
                    break

                indices = torch.LongTensor(list(range(window_start, window_end)))
                shuffled_indices = indices[torch.randperm(window_length, generator=generator)]

                current_start = i
                current_end = min(num_frames, current_start + window_length)
                if current_end == current_start + window_length:
                    # batch of frames perfectly fits the window
                    latents[:, :, current_start:current_end] = latents[:, :, shuffled_indices]
                else:
                    # handle the case where the last batch of frames does not fit perfectly with the window
                    prefix_length = current_end - current_start
                    shuffled_indices = shuffled_indices[:prefix_length]
                    latents[:, :, current_start:current_end] = latents[:, :, shuffled_indices]

        elif self._free_noise_noise_type == "repeat_context":
            num_repeats = (num_frames + self._free_noise_context_length - 1) // self._free_noise_context_length
            latents = torch.cat([latents] * num_repeats, dim=2)

        latents = latents[:, :, :num_frames]
        return latents

    def _lerp(
        self, start_index: int, end_index: int, start_tensor: torch.Tensor, end_tensor: torch.Tensor
    ) -> torch.Tensor:
        num_indices = end_index - start_index + 1
        interpolated_tensors = []

        for i in range(num_indices):
            alpha = i / (num_indices - 1)
            interpolated_tensor = (1 - alpha) * start_tensor + alpha * end_tensor
            interpolated_tensors.append(interpolated_tensor)

        interpolated_tensors = torch.cat(interpolated_tensors)
        return interpolated_tensors

    def enable_free_noise(
        self,
        context_length: Optional[int] = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
        noise_type: str = "shuffle_context",
        prompt_interpolation_callback: Optional[
            Callable[[DiffusionPipeline, int, int, torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        r"""
        Enable long video generation using FreeNoise.

        Args:
            context_length (`int`, defaults to `16`, *optional*):
                The number of video frames to process at once. It's recommended to set this to the maximum frames the
                Motion Adapter was trained with (usually 16/24/32). If `None`, the default value from the motion
                adapter config is used.
            context_stride (`int`, *optional*):
                Long videos are generated by processing many frames. FreeNoise processes these frames in sliding
                windows of size `context_length`. Context stride allows you to specify how many frames to skip between
                each window. For example, a context length of 16 and context stride of 4 would process 24 frames as:
                    [0, 15], [4, 19], [8, 23] (0-based indexing)
            weighting_scheme (`str`, defaults to `pyramid`):
                Weighting scheme for averaging latents after accumulation in FreeNoise blocks. The following weighting
                schemes are supported currently:
                    - "flat"
                       Performs weighting averaging with a flat weight pattern: [1, 1, 1, 1, 1].
                    - "pyramid"
                        Performs weighted averaging with a pyramid like weight pattern: [1, 2, 3, 2, 1].
                    - "delayed_reverse_sawtooth"
                        Performs weighted averaging with low weights for earlier frames and high-to-low weights for
                        later frames: [0.01, 0.01, 3, 2, 1].
            noise_type (`str`, defaults to "shuffle_context"):
                Must be one of ["shuffle_context", "repeat_context", "random"].
                    - "shuffle_context"
                        Shuffles a fixed batch of `context_length` latents to create a final latent of size
                        `num_frames`. This is usually the best setting for most generation scenarious. However, there
                        might be visible repetition noticeable in the kinds of motion/animation generated.
                    - "repeated_context"
                        Repeats a fixed batch of `context_length` latents to create a final latent of size
                        `num_frames`.
                    - "random"
                        The final latents are random without any repetition.
        """

        allowed_weighting_scheme = ["flat", "pyramid", "delayed_reverse_sawtooth"]
        allowed_noise_type = ["shuffle_context", "repeat_context", "random"]

        if context_length > self.motion_adapter.config.motion_max_seq_length:
            logger.warning(
                f"You have set {context_length=} which is greater than {self.motion_adapter.config.motion_max_seq_length=}. This can lead to bad generation results."
            )
        if weighting_scheme not in allowed_weighting_scheme:
            raise ValueError(
                f"The parameter `weighting_scheme` must be one of {allowed_weighting_scheme}, but got {weighting_scheme=}"
            )
        if noise_type not in allowed_noise_type:
            raise ValueError(f"The parameter `noise_type` must be one of {allowed_noise_type}, but got {noise_type=}")

        self._free_noise_context_length = context_length or self.motion_adapter.config.motion_max_seq_length
        self._free_noise_context_stride = context_stride
        self._free_noise_weighting_scheme = weighting_scheme
        self._free_noise_noise_type = noise_type
        self._free_noise_prompt_interpolation_callback = prompt_interpolation_callback or self._lerp

        if hasattr(self.unet.mid_block, "motion_modules"):
            blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        else:
            blocks = [*self.unet.down_blocks, *self.unet.up_blocks]

        for block in blocks:
            self._enable_free_noise_in_block(block)

    def disable_free_noise(self) -> None:
        r"""Disable the FreeNoise sampling mechanism."""
        self._free_noise_context_length = None

        if hasattr(self.unet.mid_block, "motion_modules"):
            blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        else:
            blocks = [*self.unet.down_blocks, *self.unet.up_blocks]

        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        for block in blocks:
            self._disable_free_noise_in_block(block)

    @property
    def free_noise_enabled(self):
        return hasattr(self, "_free_noise_context_length") and self._free_noise_context_length is not None
