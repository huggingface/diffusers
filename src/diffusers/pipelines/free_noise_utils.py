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

from typing import Optional, Union

import torch

from ..models.attention import BasicTransformerBlock, FreeNoiseTransformerBlock
from ..models.unets.unet_motion_model import (
    CrossAttnDownBlockMotion,
    DownBlockMotion,
    UpBlockMotion,
)
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

    def enable_free_noise(
        self,
        context_length: Optional[int] = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
        noise_type: str = "shuffle_context",
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
                    - "pyramid"
                        Peforms weighted averaging with a pyramid like weight pattern: [1, 2, 3, 2, 1].
            noise_type (`str`, defaults to "shuffle_context"):
                TODO
        """

        allowed_weighting_scheme = ["pyramid"]
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

        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        for block in blocks:
            self._enable_free_noise_in_block(block)

    def disable_free_noise(self) -> None:
        self._free_noise_context_length = None

        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        for block in blocks:
            self._disable_free_noise_in_block(block)

    @property
    def free_noise_enabled(self):
        return hasattr(self, "_free_noise_context_length") and self._free_noise_context_length is not None
