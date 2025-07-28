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
from typing import Tuple, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_FBC_LEADER_BLOCK_HOOK = "fbc_leader_block_hook"
_FBC_BLOCK_HOOK = "fbc_block_hook"


@dataclass
class FirstBlockCacheConfig:
    r"""
    Configuration for [First Block
    Cache](https://github.com/chengzeyi/ParaAttention/blob/7a266123671b55e7e5a2fe9af3121f07a36afc78/README.md#first-block-cache-our-dynamic-caching).

    Args:
        threshold (`float`, defaults to `0.05`):
            The threshold to determine whether or not a forward pass through all layers of the model is required. A
            higher threshold usually results in a forward pass through a lower number of layers and faster inference,
            but might lead to poorer generation quality. A lower threshold may not result in significant generation
            speedup. The threshold is compared against the absmean difference of the residuals between the current and
            cached outputs from the first transformer block. If the difference is below the threshold, the forward pass
            is skipped.
    """

    threshold: float = 0.05


class FBCSharedBlockState(BaseState):
    def __init__(self) -> None:
        super().__init__()

        self.head_block_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.head_block_residual: torch.Tensor = None
        self.tail_block_residuals: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.should_compute: bool = True

    def reset(self):
        self.tail_block_residuals = None
        self.should_compute = True


class FBCHeadBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, threshold: float):
        self.state_manager = state_manager
        self.threshold = threshold
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)

        output = self.fn_ref.original_forward(*args, **kwargs)
        is_output_tuple = isinstance(output, tuple)

        if is_output_tuple:
            hidden_states_residual = output[self._metadata.return_hidden_states_index] - original_hidden_states
        else:
            hidden_states_residual = output - original_hidden_states

        shared_state: FBCSharedBlockState = self.state_manager.get_state()
        hidden_states = encoder_hidden_states = None
        should_compute = self._should_compute_remaining_blocks(hidden_states_residual)
        shared_state.should_compute = should_compute

        if not should_compute:
            # Apply caching
            if is_output_tuple:
                hidden_states = (
                    shared_state.tail_block_residuals[0] + output[self._metadata.return_hidden_states_index]
                )
            else:
                hidden_states = shared_state.tail_block_residuals[0] + output

            if self._metadata.return_encoder_hidden_states_index is not None:
                assert is_output_tuple
                encoder_hidden_states = (
                    shared_state.tail_block_residuals[1] + output[self._metadata.return_encoder_hidden_states_index]
                )

            if is_output_tuple:
                return_output = [None] * len(output)
                return_output[self._metadata.return_hidden_states_index] = hidden_states
                return_output[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return_output = tuple(return_output)
            else:
                return_output = hidden_states
            output = return_output
        else:
            if is_output_tuple:
                head_block_output = [None] * len(output)
                head_block_output[0] = output[self._metadata.return_hidden_states_index]
                head_block_output[1] = output[self._metadata.return_encoder_hidden_states_index]
            else:
                head_block_output = output
            shared_state.head_block_output = head_block_output
            shared_state.head_block_residual = hidden_states_residual

        return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module

    @torch.compiler.disable
    def _should_compute_remaining_blocks(self, hidden_states_residual: torch.Tensor) -> bool:
        shared_state = self.state_manager.get_state()
        if shared_state.head_block_residual is None:
            return True
        prev_hidden_states_residual = shared_state.head_block_residual
        absmean = (hidden_states_residual - prev_hidden_states_residual).abs().mean()
        prev_hidden_states_absmean = prev_hidden_states_residual.abs().mean()
        diff = (absmean / prev_hidden_states_absmean).item()
        return diff > self.threshold


class FBCBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        original_hidden_states = self._metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
        original_encoder_hidden_states = None
        if self._metadata.return_encoder_hidden_states_index is not None:
            original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                "encoder_hidden_states", args, kwargs
            )

        shared_state = self.state_manager.get_state()

        if shared_state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                hidden_states_residual = encoder_hidden_states_residual = None
                if isinstance(output, tuple):
                    hidden_states_residual = (
                        output[self._metadata.return_hidden_states_index] - shared_state.head_block_output[0]
                    )
                    encoder_hidden_states_residual = (
                        output[self._metadata.return_encoder_hidden_states_index] - shared_state.head_block_output[1]
                    )
                else:
                    hidden_states_residual = output - shared_state.head_block_output
                shared_state.tail_block_residuals = (hidden_states_residual, encoder_hidden_states_residual)
            return output

        if original_encoder_hidden_states is None:
            return_output = original_hidden_states
        else:
            return_output = [None, None]
            return_output[self._metadata.return_hidden_states_index] = original_hidden_states
            return_output[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
            return_output = tuple(return_output)
        return return_output


def apply_first_block_cache(module: torch.nn.Module, config: FirstBlockCacheConfig) -> None:
    """
    Applies [First Block
    Cache](https://github.com/chengzeyi/ParaAttention/blob/4de137c5b96416489f06e43e19f2c14a772e28fd/README.md#first-block-cache-our-dynamic-caching)
    to a given module.

    First Block Cache builds on the ideas of [TeaCache](https://huggingface.co/papers/2411.19108). It is much simpler
    to implement generically for a wide range of models and has been integrated first for experimental purposes.

    Args:
        module (`torch.nn.Module`):
            The pytorch module to apply FBCache to. Typically, this should be a transformer architecture supported in
            Diffusers, such as `CogVideoXTransformer3DModel`, but external implementations may also work.
        config (`FirstBlockCacheConfig`):
            The configuration to use for applying the FBCache method.

    Example:
        ```python
        >>> import torch
        >>> from diffusers import CogView4Pipeline
        >>> from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

        >>> pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.2))

        >>> prompt = "A photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, generator=torch.Generator().manual_seed(42)).images[0]
        >>> image.save("output.png")
        ```
    """

    state_manager = StateManager(FBCSharedBlockState, (), {})
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.debug(f"Applying FBCHeadBlockHook to '{head_block_name}'")
    _apply_fbc_head_block_hook(head_block, state_manager, config.threshold)

    for name, block in remaining_blocks:
        logger.debug(f"Applying FBCBlockHook to '{name}'")
        _apply_fbc_block_hook(block, state_manager)

    logger.debug(f"Applying FBCBlockHook to tail block '{tail_block_name}'")
    _apply_fbc_block_hook(tail_block, state_manager, is_tail=True)


def _apply_fbc_head_block_hook(block: torch.nn.Module, state_manager: StateManager, threshold: float) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCHeadBlockHook(state_manager, threshold)
    registry.register_hook(hook, _FBC_LEADER_BLOCK_HOOK)


def _apply_fbc_block_hook(block: torch.nn.Module, state_manager: StateManager, is_tail: bool = False) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCBlockHook(state_manager, is_tail)
    registry.register_hook(hook, _FBC_BLOCK_HOOK)
