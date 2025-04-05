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
from typing import Callable, List, Optional

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS, _ATTENTION_CLASSES, _FEEDFORWARD_CLASSES
from ._helpers import AttentionProcessorRegistry, TransformerBlockRegistry
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name

_LAYER_SKIP_HOOK = "layer_skip_hook"


@dataclass
class LayerSkipConfig:
    r"""
    Configuration for skipping internal transformer blocks when executing a transformer model.

    Args:
        indices (`List[int]`):
            The indices of the layer to skip. This is typically the first layer in the transformer block.
        fqn (`str`, defaults to `"auto"`):
            The fully qualified name identifying the stack of transformer blocks. Typically, this is
            `transformer_blocks`, `single_transformer_blocks`, `blocks`, `layers`, or `temporal_transformer_blocks`.
    """

    indices: List[int]
    fqn: str = "auto"
    skip_attention: bool = True
    skip_attention_scores: bool = False
    skip_ff: bool = True


class AttentionScoreSkipFunctionMode(torch.overrides.TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            value = kwargs.get("value", None)
            if value is None:
                value = args[2]
            return value
        return func(*args, **kwargs)


class AttentionProcessorSkipHook(ModelHook):
    def __init__(self, skip_processor_output_fn: Callable, skip_attention_scores: bool = False):
        self.skip_processor_output_fn = skip_processor_output_fn
        self.skip_attention_scores = skip_attention_scores

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.skip_attention_scores:
            with AttentionScoreSkipFunctionMode():
                return self.fn_ref.original_forward(*args, **kwargs)
        else:
            return self.skip_processor_output_fn(module, *args, **kwargs)


class FeedForwardSkipHook(ModelHook):
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        output = kwargs.get("hidden_states", None)
        if output is None:
            output = kwargs.get("x", None)
        if output is None and len(args) > 0:
            output = args[0]
        return output


class TransformerBlockSkipHook(ModelHook):
    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        return self._metadata.skip_block_output_fn(module, *args, **kwargs)


def apply_layer_skip(module: torch.nn.Module, config: LayerSkipConfig) -> None:
    r"""
    Apply layer skipping to internal layers of a transformer.

    Args:
        module (`torch.nn.Module`):
            The transformer model to which the layer skip hook should be applied.
        config (`LayerSkipConfig`):
            The configuration for the layer skip hook.

    Example:

    ```python
    >>> from diffusers import apply_layer_skip_hook, CogVideoXTransformer3DModel, LayerSkipConfig

    >>> transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    >>> config = LayerSkipConfig(layer_index=[10, 20], fqn="transformer_blocks")
    >>> apply_layer_skip_hook(transformer, config)
    ```
    """
    _apply_layer_skip_hook(module, config)


def _apply_layer_skip_hook(module: torch.nn.Module, config: LayerSkipConfig, name: Optional[str] = None) -> None:
    name = name or _LAYER_SKIP_HOOK

    if config.skip_attention and config.skip_attention_scores:
        raise ValueError("Cannot set both `skip_attention` and `skip_attention_scores` to True. Please choose one.")

    if config.fqn == "auto":
        for identifier in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS:
            if hasattr(module, identifier):
                config.fqn = identifier
                break
        else:
            raise ValueError(
                "Could not find a suitable identifier for the transformer blocks automatically. Please provide a valid "
                "`fqn` (fully qualified name) that identifies a stack of transformer blocks."
            )

    transformer_blocks = getattr(module, config.fqn, None)
    if transformer_blocks is None or not isinstance(transformer_blocks, torch.nn.ModuleList):
        raise ValueError(
            f"Could not find {config.fqn} in the provided module, or configured `fqn` (fully qualified name) does not identify "
            f"a `torch.nn.ModuleList`. Please provide a valid `fqn` that identifies a stack of transformer blocks."
        )
    if len(config.indices) == 0:
        raise ValueError("Layer index list is empty. Please provide a non-empty list of layer indices to skip.")

    blocks_found = False
    for i, block in enumerate(transformer_blocks):
        if i not in config.indices:
            continue
        blocks_found = True
        if config.skip_attention and config.skip_ff:
            logger.debug(f"Applying TransformerBlockSkipHook to '{config.fqn}.{i}'")
            registry = HookRegistry.check_if_exists_or_initialize(block)
            hook = TransformerBlockSkipHook()
            registry.register_hook(hook, name)
        elif config.skip_attention or config.skip_attention_scores:
            for submodule_name, submodule in block.named_modules():
                if isinstance(submodule, _ATTENTION_CLASSES) and not submodule.is_cross_attention:
                    logger.debug(f"Applying AttentionProcessorSkipHook to '{config.fqn}.{i}.{submodule_name}'")
                    output_fn = AttentionProcessorRegistry.get(submodule.processor.__class__).skip_processor_output_fn
                    registry = HookRegistry.check_if_exists_or_initialize(submodule)
                    hook = AttentionProcessorSkipHook(output_fn, config.skip_attention_scores)
                    registry.register_hook(hook, name)
        elif config.skip_ff:
            for submodule_name, submodule in block.named_modules():
                if isinstance(submodule, _FEEDFORWARD_CLASSES):
                    logger.debug(f"Applying FeedForwardSkipHook to '{config.fqn}.{i}.{submodule_name}'")
                    registry = HookRegistry.check_if_exists_or_initialize(submodule)
                    hook = FeedForwardSkipHook()
                    registry.register_hook(hook, name)
        else:
            raise ValueError(
                "At least one of `skip_attention`, `skip_attention_scores`, or `skip_ff` must be set to True."
            )

    if not blocks_found:
        raise ValueError(
            f"Could not find any transformer blocks matching the provided indices {config.indices} and "
            f"fully qualified name '{config.fqn}'. Please check the indices and fqn for correctness."
        )
