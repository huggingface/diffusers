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
from .hooks import BaseMarkedState, HookRegistry, ModelHook


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


class FBCSharedBlockState(BaseMarkedState):
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

    def __init__(self, shared_state: FBCSharedBlockState, threshold: float):
        self.shared_state = shared_state
        self.threshold = threshold
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        outputs_if_skipped = self._metadata.skip_block_output_fn(module, *args, **kwargs)

        if isinstance(outputs_if_skipped, tuple):
            original_hidden_states = outputs_if_skipped[self._metadata.return_hidden_states_index]
        else:
            original_hidden_states = outputs_if_skipped

        output = self.fn_ref.original_forward(*args, **kwargs)
        is_output_tuple = isinstance(output, tuple)

        if is_output_tuple:
            hidden_states_residual = output[self._metadata.return_hidden_states_index] - original_hidden_states
        else:
            hidden_states_residual = output - original_hidden_states

        hidden_states, encoder_hidden_states = None, None
        should_compute = self._should_compute_remaining_blocks(hidden_states_residual)
        self.shared_state.should_compute = should_compute

        if not should_compute:
            # Apply caching
            if is_output_tuple:
                hidden_states = (
                    self.shared_state.tail_block_residuals[0] + output[self._metadata.return_hidden_states_index]
                )
            else:
                hidden_states = self.shared_state.tail_block_residuals[0] + output

            if self._metadata.return_encoder_hidden_states_index is not None:
                assert is_output_tuple
                encoder_hidden_states = (
                    self.shared_state.tail_block_residuals[1]
                    + output[self._metadata.return_encoder_hidden_states_index]
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
            self.shared_state.head_block_output = head_block_output
            self.shared_state.head_block_residual = hidden_states_residual

        return output

    def reset_state(self, module):
        self.shared_state.reset()
        return module

    @torch.compiler.disable
    def _should_compute_remaining_blocks(self, hidden_states_residual: torch.Tensor) -> bool:
        if self.shared_state.head_block_residual is None:
            return True
        prev_hidden_states_residual = self.shared_state.head_block_residual
        absmean = (hidden_states_residual - prev_hidden_states_residual).abs().mean()
        prev_hidden_states_absmean = prev_hidden_states_residual.abs().mean()
        diff = (absmean / prev_hidden_states_absmean).item()
        return diff > self.threshold


class FBCBlockHook(ModelHook):
    def __init__(self, shared_state: FBCSharedBlockState, is_tail: bool = False):
        super().__init__()
        self.shared_state = shared_state
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        outputs_if_skipped = self._metadata.skip_block_output_fn(module, *args, **kwargs)
        if not isinstance(outputs_if_skipped, tuple):
            outputs_if_skipped = (outputs_if_skipped,)
        original_hidden_states = outputs_if_skipped[self._metadata.return_hidden_states_index]
        original_encoder_hidden_states = None
        if self._metadata.return_encoder_hidden_states_index is not None:
            original_encoder_hidden_states = outputs_if_skipped[self._metadata.return_encoder_hidden_states_index]

        if self.shared_state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                hidden_states_residual = encoder_hidden_states_residual = None
                if isinstance(output, tuple):
                    hidden_states_residual = (
                        output[self._metadata.return_hidden_states_index] - self.shared_state.head_block_output[0]
                    )
                    encoder_hidden_states_residual = (
                        output[self._metadata.return_encoder_hidden_states_index]
                        - self.shared_state.head_block_output[1]
                    )
                else:
                    hidden_states_residual = output - self.shared_state.head_block_output
                self.shared_state.tail_block_residuals = (hidden_states_residual, encoder_hidden_states_residual)
            return output

        output_count = len(outputs_if_skipped)
        if output_count == 1:
            return_output = original_hidden_states
        else:
            return_output = [None] * output_count
            return_output[self._metadata.return_hidden_states_index] = original_hidden_states
            return_output[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
        return return_output


def apply_first_block_cache(module: torch.nn.Module, config: FirstBlockCacheConfig) -> None:
    shared_state = FBCSharedBlockState()
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.debug(f"Applying FBCHeadBlockHook to '{head_block_name}'")
    _apply_fbc_head_block_hook(head_block, shared_state, config.threshold)

    for name, block in remaining_blocks:
        logger.debug(f"Applying FBCBlockHook to '{name}'")
        _apply_fbc_block_hook(block, shared_state)

    logger.debug(f"Applying FBCBlockHook to tail block '{tail_block_name}'")
    _apply_fbc_block_hook(tail_block, shared_state, is_tail=True)


def _apply_fbc_head_block_hook(block: torch.nn.Module, state: FBCSharedBlockState, threshold: float) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCHeadBlockHook(state, threshold)
    registry.register_hook(hook, _FBC_LEADER_BLOCK_HOOK)


def _apply_fbc_block_hook(block: torch.nn.Module, state: FBCSharedBlockState, is_tail: bool = False) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCBlockHook(state, is_tail)
    registry.register_hook(hook, _FBC_BLOCK_HOOK)
