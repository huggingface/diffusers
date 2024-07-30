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

from ..models.attention import BasicTransformerBlock
from ..models.unets.unet_motion_model import (
    CrossAttnDownBlockMotion,
    DownBlockMotion,
    FreeNoiseTransformerBlock,
    UpBlockMotion,
)


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

    def enable_free_noise(
        self,
        context_length: Optional[int] = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
        shuffle: bool = True,
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
            shuffle (`str`, defaults to `True`):
                Shuffling latents is described in Equation 9. of the paper. It is a vital in improving the video
                consistency. Without shuffling, a random batch of `num_frames` latents are created. With shuffling,
                only the first `context_length` latents are shuffled and repeated.
        """
        self._free_noise_context_length = context_length or self.motion_adapter.config.motion_max_seq_length
        self._free_noise_context_stride = context_stride
        self._free_noise_weighting_scheme = weighting_scheme
        self._free_noise_shuffle = shuffle

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
