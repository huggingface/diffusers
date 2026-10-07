# Copyright 2026 Ollin Boer Bohan and The HuggingFace Team. All rights reserved.
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
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_utils import ModelMixin
from .autoencoder_tiny import AutoencoderTinyOutput
from .vae import AutoencoderMixin, DecoderOutput


class TinyVideoDecodeCache:
    """
    Per-layer memory for decoding a video chunk by chunk with [`AutoencoderTinyVideo.decode`].

    Pass the same cache to consecutive `decode(z, cache=cache)` calls: each call decodes only the latent frames in `z`,
    continuing from the frames decoded by the previous calls, and the concatenated result is identical to decoding all
    frames in a single call. Create a new cache for every new video.
    """

    def __init__(self):
        self.memory: list[torch.Tensor] | None = None


class TinyVideoMemBlock(nn.Module):
    """
    Residual block that sees the previous frame: its input is concatenated with the block input of the frame before it
    (zeros for the very first frame of a video). Frames are stacked along the batch dimension.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU()

    def forward(
        self, hidden_states: torch.Tensor, batch_size: int, last_frame: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # memory of frame t is the block input of frame t-1; `last_frame` carries it over from the previous chunk
        frames = hidden_states.unflatten(0, (batch_size, -1))
        if last_frame is None:
            last_frame = torch.zeros_like(frames[:, :1])
        memory = torch.cat([last_frame, frames[:, :-1]], dim=1).flatten(0, 1)

        residual = hidden_states
        hidden_states = self.act(self.conv1(torch.cat([hidden_states, memory], dim=1)))
        hidden_states = self.act(self.conv2(hidden_states))
        hidden_states = self.conv3(hidden_states)
        return self.act(hidden_states + residual), frames[:, -1:]


class TinyVideoEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_downscale: bool):
        super().__init__()
        self.time_stride = 2 if time_downscale else 1
        self.time_pool = nn.Conv2d(in_channels * self.time_stride, in_channels, 1, bias=False)
        self.conv_down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.mem_blocks = nn.ModuleList([TinyVideoMemBlock(out_channels) for _ in range(3)])

    def forward(self, hidden_states: torch.Tensor, batch_size: int) -> torch.Tensor:
        # fold `time_stride` consecutive frames into the channels, then pool them with a 1x1 conv
        hidden_states = hidden_states.unflatten(0, (-1, self.time_stride)).flatten(1, 2)
        hidden_states = self.time_pool(hidden_states)
        hidden_states = self.conv_down(hidden_states)
        for mem_block in self.mem_blocks:
            hidden_states, _ = mem_block(hidden_states, batch_size)
        return hidden_states


class TinyVideoDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_upscale: bool):
        super().__init__()
        self.mem_blocks = nn.ModuleList([TinyVideoMemBlock(in_channels) for _ in range(3)])
        self.upsample = nn.Upsample(scale_factor=2)
        self.time_stride = 2 if time_upscale else 1
        self.time_grow = nn.Conv2d(in_channels, in_channels * self.time_stride, 1, bias=False)
        self.conv_out = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, batch_size: int, memory: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        new_memory = []
        for mem_block, last_frame in zip(self.mem_blocks, memory):
            hidden_states, last_frame = mem_block(hidden_states, batch_size, last_frame)
            new_memory.append(last_frame)
        hidden_states = self.upsample(hidden_states)
        # grow every frame into `time_stride` consecutive frames with a 1x1 conv
        hidden_states = self.time_grow(hidden_states)
        hidden_states = hidden_states.unflatten(1, (self.time_stride, -1)).flatten(0, 1)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states, new_memory


class TinyVideoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        time_downscale: tuple[bool, ...],
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(
            [
                TinyVideoEncoderBlock(block_out_channels[max(i - 1, 0)], block_out_channels[i], time_downscale[i])
                for i in range(len(block_out_channels))
            ]
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor, batch_size: int) -> torch.Tensor:
        hidden_states = self.act(self.conv_in(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states, batch_size)
        return self.conv_out(hidden_states)


class TinyVideoDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        time_upscale: tuple[bool, ...],
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[0], 3, padding=1)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(
            [
                TinyVideoDecoderBlock(block_out_channels[i], block_out_channels[i + 1], time_upscale[i])
                for i in range(len(block_out_channels) - 1)
            ]
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def forward(
        self, hidden_states: torch.Tensor, batch_size: int, memory: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = torch.tanh(hidden_states / 3) * 3
        hidden_states = self.act(self.conv_in(hidden_states))
        new_memory = []
        for i, block in enumerate(self.blocks):
            hidden_states, block_memory = block(hidden_states, batch_size, memory[3 * i : 3 * i + 3])
            new_memory.extend(block_memory)
        hidden_states = self.act(hidden_states)
        return self.conv_out(hidden_states), new_memory


class AutoencoderTinyVideo(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    A tiny causal video autoencoder (TAEHV, [madebyollin/taehv](https://github.com/madebyollin/taehv)) that encodes to
    and decodes from the latent space of a full video VAE — e.g. `taew2_2` for the Wan 2.2 VAE — orders of magnitude
    faster than the full model, for previews and real-time decoding. Latents are the *normalized* (roughly unit
    Gaussian) latents of the full VAE.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented for
    all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, defaults to `3`): Number of channels in the input video.
        latent_channels (`int`, defaults to `48`): Number of channels in the latent space.
        patch_size (`int`, defaults to `2`):
            Pixel-(un)shuffle factor applied to the frames before the encoder and after the decoder.
        encoder_block_out_channels (`tuple[int, ...]`, defaults to `(64, 64, 64)`):
            Output channels of the encoder blocks; each block halves the spatial size.
        decoder_block_out_channels (`tuple[int, ...]`, defaults to `(256, 128, 64, 64)`):
            Channels of the decoder: the first entry is the width after the input conv, each following block doubles
            the spatial size and outputs the next entry.
        encoder_time_downscale (`tuple[bool, ...]`, defaults to `(True, True, False)`):
            Whether each encoder block halves the number of frames.
        decoder_time_upscale (`tuple[bool, ...]`, defaults to `(False, True, True)`):
            Whether each decoder block doubles the number of frames.
    """

    _skip_keys = ["memory"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 48,
        patch_size: int = 2,
        encoder_block_out_channels: tuple[int, ...] = (64, 64, 64),
        decoder_block_out_channels: tuple[int, ...] = (256, 128, 64, 64),
        encoder_time_downscale: tuple[bool, ...] = (True, True, False),
        decoder_time_upscale: tuple[bool, ...] = (False, True, True),
    ):
        super().__init__()
        self.encoder = TinyVideoEncoder(
            in_channels * patch_size**2, latent_channels, encoder_block_out_channels, encoder_time_downscale
        )
        self.decoder = TinyVideoDecoder(
            latent_channels, in_channels * patch_size**2, decoder_block_out_channels, decoder_time_upscale
        )
        self.temporal_compression_ratio = 2 ** sum(encoder_time_downscale)
        self.temporal_upsampling_ratio = 2 ** sum(decoder_time_upscale)
        self.spatial_compression_ratio = patch_size * 2 ** len(encoder_block_out_channels)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderTinyOutput | tuple[torch.Tensor]:
        r"""
        Encode a batch of videos `[B, C, T, H, W]` in `[-1, 1]`. The frames are padded at the end, by repeating the
        last one, to a multiple of `temporal_compression_ratio`.
        """
        batch_size, _, num_frames = x.shape[:3]
        if num_frames % self.temporal_compression_ratio != 0:
            num_pad = self.temporal_compression_ratio - num_frames % self.temporal_compression_ratio
            x = torch.cat([x, x[:, :, -1:].repeat_interleave(num_pad, dim=2)], dim=2)

        frames = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        frames = F.pixel_unshuffle(frames.add(1).div(2), self.config.patch_size)
        latents = self.encoder(frames, batch_size)
        latents = latents.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (latents,)
        return AutoencoderTinyOutput(latents=latents)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True, cache: TinyVideoDecodeCache | None = None
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Decode a batch of latents `[B, C, T, h, w]` to videos in `[-1, 1]`. `T` latent frames decode to `T *
        temporal_upsampling_ratio - (temporal_upsampling_ratio - 1)` frames: the first frames produced by the decoder
        are warm-up frames and are dropped.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.
            cache (`TinyVideoDecodeCache`, *optional*):
                Decode a video chunk by chunk: pass the same cache to consecutive calls and each call decodes only the
                frames in `z`, continuing from the previous calls.
        """
        batch_size = z.shape[0]
        # a fresh cache (or no cache) starts a new video; a used one continues the previous call's video
        first_chunk = cache is None or cache.memory is None
        memory = cache.memory if not first_chunk else [None] * (3 * len(self.decoder.blocks))

        latents = z.permute(0, 2, 1, 3, 4).flatten(0, 1)
        frames, memory = self.decoder(latents, batch_size, memory)
        if cache is not None:
            cache.memory = memory

        frames = F.pixel_shuffle(frames, self.config.patch_size).clamp(0, 1).mul(2).sub(1)
        frames = frames.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        if first_chunk:
            frames = frames[:, :, self.temporal_upsampling_ratio - 1 :]

        if not return_dict:
            return (frames,)
        return DecoderOutput(sample=frames)

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input video `[B, C, T, H, W]` in `[-1, 1]`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        latents = self.encode(sample).latents
        decoded = self.decode(latents).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
