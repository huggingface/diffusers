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

from ..models.attention import FeedForward, LuminaFeedForward
from ..models.attention_processor import Attention, MochiAttention


_ATTENTION_CLASSES = (Attention, MochiAttention)
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
