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

from dataclasses import dataclass
from typing import Any, Callable, Type

from ..models.attention import BasicTransformerBlock
from ..models.attention_processor import AttnProcessor2_0
from ..models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from ..models.transformers.transformer_cogview4 import CogView4AttnProcessor, CogView4TransformerBlock
from ..models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from ..models.transformers.transformer_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenReplaceSingleTransformerBlock,
    HunyuanVideoTokenReplaceTransformerBlock,
    HunyuanVideoTransformerBlock,
)
from ..models.transformers.transformer_ltx import LTXVideoTransformerBlock
from ..models.transformers.transformer_mochi import MochiTransformerBlock
from ..models.transformers.transformer_wan import WanTransformerBlock


@dataclass
class AttentionProcessorMetadata:
    skip_processor_output_fn: Callable[[Any], Any]


@dataclass
class TransformerBlockMetadata:
    skip_block_output_fn: Callable[[Any], Any]
    return_hidden_states_index: int = None
    return_encoder_hidden_states_index: int = None


class AttentionProcessorRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_class: Type, metadata: AttentionProcessorMetadata):
        cls._registry[model_class] = metadata

    @classmethod
    def get(cls, model_class: Type) -> AttentionProcessorMetadata:
        if model_class not in cls._registry:
            raise ValueError(f"Model class {model_class} not registered.")
        return cls._registry[model_class]


class TransformerBlockRegistry:
    _registry = {}

    @classmethod
    def register(cls, model_class: Type, metadata: TransformerBlockMetadata):
        cls._registry[model_class] = metadata

    @classmethod
    def get(cls, model_class: Type) -> TransformerBlockMetadata:
        if model_class not in cls._registry:
            raise ValueError(f"Model class {model_class} not registered.")
        return cls._registry[model_class]


def _register_attention_processors_metadata():
    # AttnProcessor2_0
    AttentionProcessorRegistry.register(
        model_class=AttnProcessor2_0,
        metadata=AttentionProcessorMetadata(
            skip_processor_output_fn=_skip_proc_output_fn_Attention_AttnProcessor2_0,
        ),
    )

    # CogView4AttnProcessor
    AttentionProcessorRegistry.register(
        model_class=CogView4AttnProcessor,
        metadata=AttentionProcessorMetadata(
            skip_processor_output_fn=_skip_proc_output_fn_Attention_CogView4AttnProcessor,
        ),
    )


def _register_transformer_blocks_metadata():
    # BasicTransformerBlock
    TransformerBlockRegistry.register(
        model_class=BasicTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_BasicTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

    # CogVideoX
    TransformerBlockRegistry.register(
        model_class=CogVideoXBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_CogVideoXBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # CogView4
    TransformerBlockRegistry.register(
        model_class=CogView4TransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_CogView4TransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # Flux
    TransformerBlockRegistry.register(
        model_class=FluxTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_FluxTransformerBlock,
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=FluxSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_FluxSingleTransformerBlock,
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )

    # HunyuanVideo
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_HunyuanVideoTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_HunyuanVideoSingleTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTokenReplaceTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_HunyuanVideoTokenReplaceTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTokenReplaceSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_HunyuanVideoTokenReplaceSingleTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # LTXVideo
    TransformerBlockRegistry.register(
        model_class=LTXVideoTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_LTXVideoTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

    # Mochi
    TransformerBlockRegistry.register(
        model_class=MochiTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_MochiTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # Wan
    TransformerBlockRegistry.register(
        model_class=WanTransformerBlock,
        metadata=TransformerBlockMetadata(
            skip_block_output_fn=_skip_block_output_fn_WanTransformerBlock,
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )


# fmt: off
def _skip_attention___ret___hidden_states(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", None)
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
    return hidden_states


def _skip_attention___ret___hidden_states___encoder_hidden_states(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", None)
    encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
    if encoder_hidden_states is None and len(args) > 1:
        encoder_hidden_states = args[1]
    return hidden_states, encoder_hidden_states


_skip_proc_output_fn_Attention_AttnProcessor2_0 = _skip_attention___ret___hidden_states
_skip_proc_output_fn_Attention_CogView4AttnProcessor = _skip_attention___ret___hidden_states___encoder_hidden_states


def _skip_block_output_fn___hidden_states_0___ret___hidden_states(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", None)
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
    return hidden_states


def _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", None)
    encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
    if encoder_hidden_states is None and len(args) > 1:
        encoder_hidden_states = args[1]
    return hidden_states, encoder_hidden_states


def _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___encoder_hidden_states___hidden_states(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", None)
    encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
    if hidden_states is None and len(args) > 0:
        hidden_states = args[0]
    if encoder_hidden_states is None and len(args) > 1:
        encoder_hidden_states = args[1]
    return encoder_hidden_states, hidden_states


_skip_block_output_fn_BasicTransformerBlock = _skip_block_output_fn___hidden_states_0___ret___hidden_states
_skip_block_output_fn_CogVideoXBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_CogView4TransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_FluxTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___encoder_hidden_states___hidden_states
_skip_block_output_fn_FluxSingleTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___encoder_hidden_states___hidden_states
_skip_block_output_fn_HunyuanVideoTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_HunyuanVideoSingleTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_HunyuanVideoTokenReplaceTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_HunyuanVideoTokenReplaceSingleTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_LTXVideoTransformerBlock = _skip_block_output_fn___hidden_states_0___ret___hidden_states
_skip_block_output_fn_MochiTransformerBlock = _skip_block_output_fn___hidden_states_0___encoder_hidden_states_1___ret___hidden_states___encoder_hidden_states
_skip_block_output_fn_WanTransformerBlock = _skip_block_output_fn___hidden_states_0___ret___hidden_states
# fmt: on


_register_attention_processors_metadata()
_register_transformer_blocks_metadata()
