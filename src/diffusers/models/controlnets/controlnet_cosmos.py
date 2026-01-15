from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..modeling_utils import ModelMixin
from ..transformers.transformer_cosmos import CosmosPatchEmbed
from .controlnet import zero_module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CosmosControlNetOutput(BaseOutput):
    block_controlnet_hidden_states: Tuple[torch.Tensor]


class CosmosControlNetBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = zero_module(nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


# TODO(migmartin): implement me
# see i4/projects/cosmos/transfer2/networks/minimal_v4_lvg_dit_control_vace.py
class CosmosControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Minimal ControlNet for Cosmos Transfer2.5.

    This module projects encoded control latents into per-block residuals aligned with the
    `CosmosTransformer3DModel` hidden size. All projections are zero-initialized so the ControlNet
    starts neutral by default.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 4,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        control_block_indices: Tuple[int, ...] = (6, 13, 20, 27),
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.patch_embed = CosmosPatchEmbed(in_channels, hidden_size, patch_size, bias=False)
        self.control_blocks = nn.ModuleList(
            CosmosControlNetBlock(hidden_size) for _ in range(num_layers)
        )

    def _expand_conditioning_scale(self, conditioning_scale: Union[float, List[float]]) -> List[float]:
        if isinstance(conditioning_scale, list):
            scales = conditioning_scale
        else:
            scales = [conditioning_scale] * len(self.control_blocks)

        if len(scales) != len(self.control_blocks):
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
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        conditioning_scale: Union[float, List[float]] = 1.0,
        return_dict: bool = True,
    ) -> Union[Tuple[Tuple[torch.Tensor, ...]], CosmosControlNetOutput]:
        del hidden_states, timestep, encoder_hidden_states  # not used in this minimal control path

        control_hidden_states = self.patch_embed(controlnet_cond)
        control_hidden_states = control_hidden_states.flatten(1, 3)

        scales = self._expand_conditioning_scale(conditioning_scale)
        control_residuals = tuple(block(control_hidden_states) * scale for block, scale in zip(self.control_blocks, scales))

        if not return_dict:
            return (control_residuals,)

        return CosmosControlNetOutput(block_controlnet_hidden_states=control_residuals)
