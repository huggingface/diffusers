from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..modeling_utils import ModelMixin
from ..transformers.transformer_cosmos import (
    CosmosPatchEmbed,
    CosmosTransformerBlock,
)
from .controlnet import zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(migmartin): implement me
# see i4/projects/cosmos/transfer2/networks/minimal_v4_lvg_dit_control_vace.py
class CosmosControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    ControlNet for Cosmos Transfer2.5.
    """

    @register_to_config
    def __init__(
        self,
        n_controlnet_blocks: int = 4,
        in_channels: int = 16,
        model_channels: int = 2048,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.patch_embed = CosmosPatchEmbed(in_channels, model_channels, patch_size, bias=False)
        self.control_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                    img_context=True,
                    before_proj=(block_idx == 0),
                    after_proj=True,
                )
                for block_idx in range(n_controlnet_blocks)
            ]
        )

    def _expand_conditioning_scale(self, conditioning_scale: Union[float, List[float]]) -> List[float]:
        if isinstance(conditioning_scale, list):
            scales = conditioning_scale
        else:
            scales = [conditioning_scale] * len(self.control_blocks)

        if len(scales) < len(self.control_blocks):
            logger.warning(
                "Received %d control scales, but control network defines %d blocks. "
                "Scales will be trimmed or repeated to match.",
                len(scales),
                len(self.control_blocks),
            )
            scales = (scales * len(self.control_blocks))[: len(self.control_blocks)]
        return scales

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: Union[float, List[float]] = 1.0,
        temb: Optional[torch.Tensor] = None,
        embedded_timestep: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        encoder_hidden_states: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        # TODO: check if temb, etc. is None
        # if so, then do our own embedding of the inputs

        control_hidden_states = self.patch_embed(controlnet_cond)
        control_hidden_states = control_hidden_states.flatten(1, 3)

        scales = self._expand_conditioning_scale(conditioning_scale)
        result = []
        for block, scale in zip(self.control_blocks, scales):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
                controlnet_residual=None,
            )
            result.append(hidden_states * scale)
        return result
