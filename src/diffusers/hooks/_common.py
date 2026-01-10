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

from typing import Optional

import torch

from ..models.attention import AttentionModuleMixin, FeedForward, LuminaFeedForward
from ..models.attention_processor import Attention, MochiAttention


_ATTENTION_CLASSES = (Attention, MochiAttention, AttentionModuleMixin)
_FEEDFORWARD_CLASSES = (FeedForward, LuminaFeedForward)

_SPATIAL_TRANSFORMER_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks", "single_transformer_blocks", "layers")
_TEMPORAL_TRANSFORMER_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks",)
_CROSS_TRANSFORMER_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks", "layers")

_ALL_TRANSFORMER_BLOCK_IDENTIFIERS = tuple(
    {
        *_SPATIAL_TRANSFORMER_BLOCK_IDENTIFIERS,
        *_TEMPORAL_TRANSFORMER_BLOCK_IDENTIFIERS,
        *_CROSS_TRANSFORMER_BLOCK_IDENTIFIERS,
    }
)

# Layers supported for group offloading and layerwise casting
_GO_LC_SUPPORTED_PYTORCH_LAYERS = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
    # TODO(aryan): look into torch.nn.LayerNorm, torch.nn.GroupNorm later, seems to be causing some issues with CogVideoX
    # because of double invocation of the same norm layer in CogVideoXLayerNorm
)


def _get_submodule_from_fqn(module: torch.nn.Module, fqn: str) -> Optional[torch.nn.Module]:
    for submodule_name, submodule in module.named_modules():
        if submodule_name == fqn:
            return submodule
    return None
