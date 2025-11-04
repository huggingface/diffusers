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

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type


@dataclass
class AttentionProcessorMetadata:
    skip_processor_output_fn: Callable[[Any], Any]


@dataclass
class TransformerBlockMetadata:
    return_hidden_states_index: int = None
    return_encoder_hidden_states_index: int = None

    _cls: Type = None
    _cached_parameter_indices: Dict[str, int] = None

    def _get_parameter_from_args_kwargs(self, identifier: str, args=(), kwargs=None):
        kwargs = kwargs or {}
        if identifier in kwargs:
            return kwargs[identifier]
        if self._cached_parameter_indices is not None:
            return args[self._cached_parameter_indices[identifier]]
        if self._cls is None:
            raise ValueError("Model class is not set for metadata.")
        parameters = list(inspect.signature(self._cls.forward).parameters.keys())
        parameters = parameters[1:]  # skip `self`
        self._cached_parameter_indices = {param: i for i, param in enumerate(parameters)}
        if identifier not in self._cached_parameter_indices:
            raise ValueError(f"Parameter '{identifier}' not found in function signature but was requested.")
        index = self._cached_parameter_indices[identifier]
        if index >= len(args):
            raise ValueError(f"Expected {index} arguments but got {len(args)}.")
        return args[index]


class AttentionProcessorRegistry:
    _registry = {}
    # TODO(aryan): this is only required for the time being because we need to do the registrations
    # for classes. If we do it eagerly, i.e. call the functions in global scope, we will get circular
    # import errors because of the models imported in this file.
    _is_registered = False

    @classmethod
    def register(cls, model_class: Type, metadata: AttentionProcessorMetadata):
        cls._register()
        cls._registry[model_class] = metadata

    @classmethod
    def get(cls, model_class: Type) -> AttentionProcessorMetadata:
        cls._register()
        if model_class not in cls._registry:
            raise ValueError(f"Model class {model_class} not registered.")
        return cls._registry[model_class]

    @classmethod
    def _register(cls):
        if cls._is_registered:
            return
        cls._is_registered = True
        _register_attention_processors_metadata()


class TransformerBlockRegistry:
    _registry = {}
    # TODO(aryan): this is only required for the time being because we need to do the registrations
    # for classes. If we do it eagerly, i.e. call the functions in global scope, we will get circular
    # import errors because of the models imported in this file.
    _is_registered = False

    @classmethod
    def register(cls, model_class: Type, metadata: TransformerBlockMetadata):
        cls._register()
        metadata._cls = model_class
        cls._registry[model_class] = metadata

    @classmethod
    def get(cls, model_class: Type) -> TransformerBlockMetadata:
        cls._register()
        if model_class not in cls._registry:
            raise ValueError(f"Model class {model_class} not registered.")
        return cls._registry[model_class]

    @classmethod
    def _register(cls):
        if cls._is_registered:
            return
        cls._is_registered = True
        _register_transformer_blocks_metadata()


def _register_attention_processors_metadata():
    from ..models.attention_processor import AttnProcessor2_0
    from ..models.transformers.transformer_cogview4 import CogView4AttnProcessor
    from ..models.transformers.transformer_flux import FluxAttnProcessor
    from ..models.transformers.transformer_hunyuanimage import HunyuanImageAttnProcessor
    from ..models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0
    from ..models.transformers.transformer_wan import WanAttnProcessor2_0

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

    # WanAttnProcessor2_0
    AttentionProcessorRegistry.register(
        model_class=WanAttnProcessor2_0,
        metadata=AttentionProcessorMetadata(
            skip_processor_output_fn=_skip_proc_output_fn_Attention_WanAttnProcessor2_0,
        ),
    )

    # FluxAttnProcessor
    AttentionProcessorRegistry.register(
        model_class=FluxAttnProcessor,
        metadata=AttentionProcessorMetadata(skip_processor_output_fn=_skip_proc_output_fn_Attention_FluxAttnProcessor),
    )

    # QwenDoubleStreamAttnProcessor2
    AttentionProcessorRegistry.register(
        model_class=QwenDoubleStreamAttnProcessor2_0,
        metadata=AttentionProcessorMetadata(
            skip_processor_output_fn=_skip_proc_output_fn_Attention_QwenDoubleStreamAttnProcessor2_0
        ),
    )

    # HunyuanImageAttnProcessor
    AttentionProcessorRegistry.register(
        model_class=HunyuanImageAttnProcessor,
        metadata=AttentionProcessorMetadata(
            skip_processor_output_fn=_skip_proc_output_fn_Attention_HunyuanImageAttnProcessor,
        ),
    )


def _register_transformer_blocks_metadata():
    from ..models.attention import BasicTransformerBlock
    from ..models.transformers.cogvideox_transformer_3d import CogVideoXBlock
    from ..models.transformers.transformer_bria import BriaTransformerBlock
    from ..models.transformers.transformer_cogview4 import CogView4TransformerBlock
    from ..models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
    from ..models.transformers.transformer_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTokenReplaceSingleTransformerBlock,
        HunyuanVideoTokenReplaceTransformerBlock,
        HunyuanVideoTransformerBlock,
    )
    from ..models.transformers.transformer_hunyuanimage import (
        HunyuanImageSingleTransformerBlock,
        HunyuanImageTransformerBlock,
    )
    from ..models.transformers.transformer_ltx import LTXVideoTransformerBlock
    from ..models.transformers.transformer_mochi import MochiTransformerBlock
    from ..models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    from ..models.transformers.transformer_wan import WanTransformerBlock

    # BasicTransformerBlock
    TransformerBlockRegistry.register(
        model_class=BasicTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=BriaTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

    # CogVideoX
    TransformerBlockRegistry.register(
        model_class=CogVideoXBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # CogView4
    TransformerBlockRegistry.register(
        model_class=CogView4TransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # Flux
    TransformerBlockRegistry.register(
        model_class=FluxTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=FluxSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )

    # HunyuanVideo
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTokenReplaceTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanVideoTokenReplaceSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # LTXVideo
    TransformerBlockRegistry.register(
        model_class=LTXVideoTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

    # Mochi
    TransformerBlockRegistry.register(
        model_class=MochiTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )

    # Wan
    TransformerBlockRegistry.register(
        model_class=WanTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )

    # QwenImage
    TransformerBlockRegistry.register(
        model_class=QwenImageTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
        ),
    )

    # HunyuanImage2.1
    TransformerBlockRegistry.register(
        model_class=HunyuanImageTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=HunyuanImageSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
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
_skip_proc_output_fn_Attention_WanAttnProcessor2_0 = _skip_attention___ret___hidden_states
# not sure what this is yet.
_skip_proc_output_fn_Attention_FluxAttnProcessor = _skip_attention___ret___hidden_states
_skip_proc_output_fn_Attention_QwenDoubleStreamAttnProcessor2_0 = _skip_attention___ret___hidden_states
_skip_proc_output_fn_Attention_HunyuanImageAttnProcessor = _skip_attention___ret___hidden_states
# fmt: on
