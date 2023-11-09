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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..schedulers import ConsistencyDecoderScheduler
from ..utils import BaseOutput
from ..utils.accelerate_utils import apply_forward_hook
from ..utils.torch_utils import randn_tensor
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from .modeling_utils import ModelMixin
from .unet_2d import UNet2DModel
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder


@dataclass
class ConsistencyDecoderVAEOutput(BaseOutput):
    """
    Output of encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class ConsistencyDecoderVAE(ModelMixin, ConfigMixin):
    r"""
    The consistency decoder used with DALL-E 3.

    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline, ConsistencyDecoderVAE

        >>> vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=pipe.torch_dtype)
        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", vae=vae, torch_dtype=torch.float16
        ... ).to("cuda")

        >>> pipe("horse", generator=torch.manual_seed(0)).images
        ```
    """

    @register_to_config
    def __init__(
        self,
        scaling_factor=0.18215,
        latent_channels=4,
        encoder_act_fn="silu",
        encoder_block_out_channels=(128, 256, 512, 512),
        encoder_double_z=True,
        encoder_down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        encoder_in_channels=3,
        encoder_layers_per_block=2,
        encoder_norm_num_groups=32,
        encoder_out_channels=4,
        decoder_add_attention=False,
        decoder_block_out_channels=(320, 640, 1024, 1024),
        decoder_down_block_types=(
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
        ),
        decoder_downsample_padding=1,
        decoder_in_channels=7,
        decoder_layers_per_block=3,
        decoder_norm_eps=1e-05,
        decoder_norm_num_groups=32,
        decoder_num_train_timesteps=1024,
        decoder_out_channels=6,
        decoder_resnet_time_scale_shift="scale_shift",
        decoder_time_embedding_type="learned",
        decoder_up_block_types=(
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
        ),
    ):
        super().__init__()
        self.encoder = Encoder(
            act_fn=encoder_act_fn,
            block_out_channels=encoder_block_out_channels,
            double_z=encoder_double_z,
            down_block_types=encoder_down_block_types,
            in_channels=encoder_in_channels,
            layers_per_block=encoder_layers_per_block,
            norm_num_groups=encoder_norm_num_groups,
            out_channels=encoder_out_channels,
        )

        self.decoder_unet = UNet2DModel(
            add_attention=decoder_add_attention,
            block_out_channels=decoder_block_out_channels,
            down_block_types=decoder_down_block_types,
            downsample_padding=decoder_downsample_padding,
            in_channels=decoder_in_channels,
            layers_per_block=decoder_layers_per_block,
            norm_eps=decoder_norm_eps,
            norm_num_groups=decoder_norm_num_groups,
            num_train_timesteps=decoder_num_train_timesteps,
            out_channels=decoder_out_channels,
            resnet_time_scale_shift=decoder_resnet_time_scale_shift,
            time_embedding_type=decoder_time_embedding_type,
            up_block_types=decoder_up_block_types,
        )
        self.decoder_scheduler = ConsistencyDecoderScheduler()
        self.register_to_config(block_out_channels=encoder_block_out_channels)
        self.register_buffer(
            "means",
            torch.tensor([0.38862467, 0.02253063, 0.07381133, -0.0171294])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "stds", torch.tensor([0.9654121, 1.0440036, 0.76147926, 0.77022034])[None, :, None, None], persistent=False
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.enable_tiling
    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.disable_tiling
    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.enable_slicing
    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.disable_slicing
    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
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
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
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

        self.set_attn_processor(processor, _remove_lora=True)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[ConsistencyDecoderVAEOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.consistecy_decoder_vae.ConsistencyDecoderOoutput`] instead of a plain
                tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return ConsistencyDecoderVAEOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        num_inference_steps=2,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = (z * self.config.scaling_factor - self.means) / self.stds

        scale_factor = 2 ** (len(self.config.block_out_channels) - 1)
        z = F.interpolate(z, mode="nearest", scale_factor=scale_factor)

        batch_size, _, height, width = z.shape

        self.decoder_scheduler.set_timesteps(num_inference_steps, device=self.device)

        x_t = self.decoder_scheduler.init_noise_sigma * randn_tensor(
            (batch_size, 3, height, width), generator=generator, dtype=z.dtype, device=z.device
        )

        for t in self.decoder_scheduler.timesteps:
            model_input = torch.concat([self.decoder_scheduler.scale_model_input(x_t, t), z], dim=1)
            model_output = self.decoder_unet(model_input, t).sample[:, :3, :, :]
            prev_sample = self.decoder_scheduler.step(model_output, t, x_t, generator).prev_sample
            x_t = prev_sample

        x_0 = x_t

        if not return_dict:
            return (x_0,)

        return DecoderOutput(sample=x_0)

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.blend_v
    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    # Copied from diffusers.models.autoencoder_kl.AutoencoderKL.blend_h
    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> ConsistencyDecoderVAEOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] instead of a
                plain tuple.

        Returns:
            [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] or `tuple`:
                If return_dict is True, a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned,
                otherwise a plain `tuple` is returned.
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

        return ConsistencyDecoderVAEOutput(latent_dist=posterior)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
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
        dec = self.decode(z, generator=generator).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
