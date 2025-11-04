# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import deprecate, logging
from ...utils.accelerate_utils import apply_forward_hook
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AutoencoderKL(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision in which case `force_upcast`
            can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False
        self.use_dp = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def enable_dp(
        self,
        world_size: Optional[int] = None,
        hw_splits: Optional[Tuple[int, int]] = None,
        overlap_ratio: Optional[float] = None,
        overlap_pixels: Optional[int] = None
    ) -> None:
        r"""
        """
        if world_size is None:
            world_size = dist.get_world_size()

        if world_size <= 1 or world_size > dist.get_world_size():
            logger.warning(
                f"Supported world_size for vae dp is between 2 - {dist.get_world_size}, but got {world_size}. " \
                f"Fall back to normal vae")
            return

        if hw_splits is None:
            hw_splits = (1, int(world_size))

        assert len(hw_splits) == 2, f"'hw_splits' should be a tuple of 2 int, but got length {len(hw_splits)}"

        h_split, w_split = map(int, hw_splits)

        self.use_dp = True
        self.h_split, self.w_split = h_split, w_split
        self.world_size = world_size
        self.overlap_ratio = overlap_ratio
        self.overlap_pixels = overlap_pixels
        self.spatial_compression_ratio = 2 ** (len(self.config.block_out_channels) - 1)

        dp_ranks = list(range(0, world_size))
        self.vae_dp_group = dist.new_group(ranks=dp_ranks)
        self.rank = dist.get_rank()
        # patch_ranks_flatten = [tile_idx % world_size for tile_idx in range(num_tiles)]
        # self.patch_ranks = torch.Tensor(patch_ranks_flatten).reshape(h_split, w_split)
        self.tile_idxs_per_rank = [[] for _ in range(self.world_size)]
        self.num_tiles_per_rank = [0] * self.world_size
        rank_idx = 0
        for h_idx in range(self.h_split):
            for w_idx in range(self.w_split):
                rank_idx %= self.world_size
                self.tile_idxs_per_rank[rank_idx].append((h_idx, w_idx))
                self.num_tiles_per_rank[rank_idx] += 1
                rank_idx += 1

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_dp:
            return self._tiled_encode(x)
        if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
            return self._tiled_encode(x)

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_dp:
            return self.tiled_decode_with_dp(z, return_dict=return_dict)
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        enc = torch.cat(result_rows, dim=2)
        return enc

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        deprecation_message = (
            "The tiled_encode implementation supporting the `return_dict` parameter is deprecated. In the future, the "
            "implementation of this method will be replaced with that of `_tiled_encode` and you will no longer be able "
            "to pass `return_dict`. You will also have to create a `DiagonalGaussianDistribution()` from the returned value."
        )
        deprecate("tiled_encode", "1.0.0", deprecation_message, standard_warn=False)

        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                if self.config.use_post_quant_conv:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _tiled_encode_with_dp(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """

        _, _, height, width = x.shape
        device = x.device
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        # Calculate stride based on h_split and w_split
        tile_latent_stride_height = int((latent_height + self.h_split - 1) / self.h_split)
        tile_latent_stride_width = int((latent_width + self.w_split - 1) / self.w_split)

        # Calculate overlap in latent space
        overlap_latent_height = 3
        overlap_latent_width = 3
        if self.overlap_pixels is not None:
            overlap_latent = (self.overlap_pixels + self.spatial_compression_ratio - 1) // self.spatial_compression_ratio
            overlap_latent_height = overlap_latent
            overlap_latent_width = overlap_latent
        elif self.overlap_ratio is not None:
            overlap_latent_height = int(self.overlap_ratio * latent_height)
            overlap_latent_width = int(self.overlap_ratio * latent_width)

        # Calculate minimum tile size in latent space
        tile_latent_min_height = tile_latent_stride_height + overlap_latent_height
        tile_latent_min_width = tile_latent_stride_width + overlap_latent_width

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tile_sample_min_height = tile_latent_min_height * self.spatial_compression_ratio
        tile_sample_min_width = tile_latent_min_width * self.spatial_compression_ratio
        tile_sample_stride_height = tile_latent_stride_height * self.spatial_compression_ratio
        tile_sample_stride_width = tile_latent_stride_width * self.spatial_compression_ratio

        # Determine tile grid dimensions - patch_ranks shape is [h_split, w_split]
        num_tile_rows = self.h_split
        num_tile_cols = self.w_split

        local_tiles = []
        local_hw_shapes = []

        for h_idx, w_idx in self.tile_idxs_per_rank[self.rank]:
            patch_height_start = h_idx * tile_sample_stride_height
            patch_height_end = patch_height_start + tile_sample_min_height
            patch_width_start = w_idx * tile_sample_stride_width
            patch_width_end = patch_width_start + tile_sample_min_width

            tile = x[:, :, patch_height_start : patch_height_end, patch_width_start : patch_width_end]
            tile = self.encoder(tile)
            if self.config.use_quant_conv:
                tile = self.quant_conv(tile)

            local_tiles.append(tile.flatten(-2, -1))
            local_hw_shapes.append(torch.Tensor([*tile.shape[-2:]]).to(device).int())

        # concat all tiles on local rank
        local_tiles = torch.cat(local_tiles, dim=-1)
        local_hw_shapes = torch.stack(local_hw_shapes)

        # get all hw shapes for each rank (perhaps has different shapes for last tile)
        gathered_shape_list = [torch.empty((num_tiles, 2), dtype=local_hw_shapes.dtype, device=device) 
                        for num_tiles in self.num_tiles_per_rank]
        dist.all_gather(gathered_shape_list, local_hw_shapes, group=self.vae_dp_group)

        # gather tiles on all ranks
        bc_ = local_tiles.shape[:-1]
        gathered_tiles = [
            torch.empty(
                (*bc_, tiles_shape.prod(dim=1).sum().item()), 
                dtype=local_tiles.dtype, device=device) for tiles_shape in gathered_shape_list
        ]
        dist.all_gather(gathered_tiles, local_tiles, group=self.vae_dp_group)

        # put tiles in rows based on tile_idxs_per_rank
        rows = [[None] * num_tile_cols for _ in range(num_tile_rows)]
        for rank_idx, tile_idxs in enumerate(self.tile_idxs_per_rank):
            if not tile_idxs:
                continue
            rank_tile_hw_shapes = gathered_shape_list[rank_idx]
            hw_start_idx = 0
            # perhaps has more than one tile in each rank, get each by hw_shapes
            for tile_idx, (h_idx, w_idx) in enumerate(tile_idxs):
                rank_tile_hw_shape = rank_tile_hw_shapes[tile_idx]
                hw_end_idx = hw_start_idx + rank_tile_hw_shape.prod().item() # flattend hw
                rows[h_idx][w_idx] = gathered_tiles[rank_idx][..., hw_start_idx:hw_end_idx].unflatten(
                    -1, rank_tile_hw_shape.tolist()) # unflatten hw dim
                hw_start_idx = hw_end_idx
 
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=3))

        enc = torch.cat(result_rows, dim=2)[:, :, :latent_height, :latent_width]
        return enc

    def tiled_decode_with_dp(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        _, _, height, width = z.shape
        device = z.device
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        # Calculate stride based on h_split and w_split
        tile_latent_stride_height = int((height + self.h_split - 1) / self.h_split)
        tile_latent_stride_width = int((width + self.w_split - 1) / self.w_split)

        # Calculate overlap in latent space
        overlap_latent_height = 3
        overlap_latent_width = 3
        if self.overlap_pixels is not None:
            overlap_latent = (self.overlap_pixels + self.spatial_compression_ratio - 1) // self.spatial_compression_ratio
            overlap_latent_height = overlap_latent
            overlap_latent_width = overlap_latent
        elif self.overlap_ratio is not None:
            overlap_latent_height = int(self.overlap_ratio * height)
            overlap_latent_width = int(self.overlap_ratio * width)

        # Calculate minimum tile size in latent space
        tile_latent_min_height = tile_latent_stride_height + overlap_latent_height
        tile_latent_min_width = tile_latent_stride_width + overlap_latent_width

        # Convert min/stride to sample space
        tile_sample_min_height = tile_latent_min_height * self.spatial_compression_ratio
        tile_sample_min_width = tile_latent_min_width * self.spatial_compression_ratio
        tile_sample_stride_height = tile_latent_stride_height * self.spatial_compression_ratio
        tile_sample_stride_width = tile_latent_stride_width * self.spatial_compression_ratio

        blend_height = tile_sample_min_height - tile_sample_stride_height
        blend_width = tile_sample_min_width - tile_sample_stride_width

        # Determine tile grid dimensions - patch_ranks shape is [h_split, w_split]
        num_tile_rows = self.h_split
        num_tile_cols = self.w_split

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        # Each rank computes only tiles assigned to it based on tile_idxs_per_rank
        local_tiles = [] # List to store tiles computed by this rank
        local_hw_shapes = [] # List to store shapes of tiles by this rank
        
        for h_idx, w_idx in self.tile_idxs_per_rank[self.rank]:
            patch_height_start = h_idx * tile_latent_stride_height
            patch_height_end = patch_height_start + tile_latent_min_height
            patch_width_start = w_idx * tile_latent_stride_width
            patch_width_end = patch_width_start + tile_latent_min_width

            tile = z[:, :, patch_height_start : patch_height_end, patch_width_start : patch_width_end]
            if self.config.use_post_quant_conv:
                tile = self.post_quant_conv(tile)
            decoded = self.decoder(tile)

            local_tiles.append(decoded.flatten(-2, -1)) # flatten h,w dim for concate all tiles in one rank
            local_hw_shapes.append(torch.Tensor([*decoded.shape[-2:]]).to(device).int()) # record hw for futher unflatten

        # concat all tiles on local rank
        local_tiles = torch.cat(local_tiles, dim=-1)
        local_hw_shapes = torch.stack(local_hw_shapes)

        # get all hw shapes for each rank (perhaps has different shapes for last tile)
        gathered_shape_list = [torch.empty((num_tiles, 2), dtype=local_hw_shapes.dtype, device=device) 
                        for num_tiles in self.num_tiles_per_rank]
        dist.all_gather(gathered_shape_list, local_hw_shapes, group=self.vae_dp_group)

        # gather tiles on all ranks
        bcn_ = local_tiles.shape[:-1]
        gathered_tiles = [
            torch.empty(
                (*bcn_, tiles_shape.prod(dim=1).sum().item()), 
                dtype=local_tiles.dtype, device=device) for tiles_shape in gathered_shape_list
        ]
        dist.all_gather(gathered_tiles, local_tiles, group=self.vae_dp_group)

        # put tiles in rows based on tile_idxs_per_rank
        rows = [[None] * num_tile_cols for _ in range(num_tile_rows)]
        for rank_idx, tile_idxs in enumerate(self.tile_idxs_per_rank):
            if not tile_idxs:
                continue
            rank_tile_hw_shapes = gathered_shape_list[rank_idx]
            hw_start_idx = 0
            # perhaps has more than one tile in each rank, get each by hw_shapes
            for tile_idx, (h_idx, w_idx) in enumerate(tile_idxs):
                rank_tile_hw_shape = rank_tile_hw_shapes[tile_idx]
                hw_end_idx = hw_start_idx + rank_tile_hw_shape.prod().item() # flattend hw
                rows[h_idx][w_idx] = gathered_tiles[rank_idx][..., hw_start_idx:hw_end_idx].unflatten(
                    -1, rank_tile_hw_shape.tolist()) # unflatten hw dim
                hw_start_idx = hw_end_idx

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :tile_sample_stride_height, :tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)[:, :, :sample_height, :sample_width]
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
