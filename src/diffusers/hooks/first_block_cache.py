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

import inspect
from dataclasses import dataclass
from typing import Tuple, Union

import torch

from ..utils import get_logger
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from .hooks import HookRegistry, ModelHook
from .utils import _extract_return_information


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
            higher threshold usually results in lower number of forward passes and faster inference, but might lead to
            poorer generation quality. A lower threshold may not result in significant generation speedup. The
            threshold is compared against the absmean difference of the residuals between the current and cached
            outputs from the first transformer block. If the difference is below the threshold, the forward pass is
            skipped.
    """

    threshold: float = 0.05


class FBCSharedBlockState:
    def __init__(self) -> None:
        self.head_block_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.head_block_residual: torch.Tensor = None
        self.tail_block_residuals: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.should_compute: bool = True

    def reset(self):
        self.tail_block_residuals = None
        self.should_compute = True

    def __repr__(self):
        return f"FirstBlockCacheSharedState(cache={self.cache})"


class FBCHeadBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, shared_state: FBCSharedBlockState, threshold: float):
        self.shared_state = shared_state
        self.threshold = threshold

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        inputs = inspect.signature(module.__class__.forward)
        inputs_index_to_str = dict(enumerate(inputs.parameters.keys()))
        inputs_str_to_index = {v: k for k, v in inputs_index_to_str.items()}

        try:
            outputs = _extract_return_information(module.__class__.forward)
            outputs_index_to_str = dict(enumerate(outputs))
            outputs_str_to_index = {v: k for k, v in outputs_index_to_str.items()}
        except RuntimeError:
            logger.error(f"Failed to extract return information for {module.__class__}")
            raise NotImplementedError(
                f"Module {module.__class__} is not supported with FirstBlockCache. Please open an issue at "
                f"https://github.com/huggingface/diffusers to notify us about the error with a minimal example "
                f"in order for us to add support for this module."
            )

        self._inputs_index_to_str = inputs_index_to_str
        self._inputs_str_to_index = inputs_str_to_index
        self._outputs_index_to_str = outputs_index_to_str
        self._outputs_str_to_index = outputs_str_to_index
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        hs_input_idx = self._inputs_str_to_index.get("hidden_states")
        ehs_input_idx = self._inputs_str_to_index.get("encoder_hidden_states", None)
        original_hs = kwargs.get("hidden_states", None)
        original_ehs = kwargs.get("encoder_hidden_states", None)
        original_hs = original_hs if original_hs is not None else args[hs_input_idx]
        if ehs_input_idx is not None:
            original_ehs = original_ehs if original_ehs is not None else args[ehs_input_idx]

        hs_output_idx = self._outputs_str_to_index.get("hidden_states")
        ehs_output_idx = self._outputs_str_to_index.get("encoder_hidden_states", None)
        assert (ehs_input_idx is None) == (ehs_output_idx is None)

        output = self.fn_ref.original_forward(*args, **kwargs)

        hs_residual = None
        if isinstance(output, tuple):
            hs_residual = output[hs_output_idx] - original_hs
        else:
            hs_residual = output - original_hs

        should_compute = self._should_compute_remaining_blocks(hs_residual)
        self.shared_state.should_compute = should_compute

        hs, ehs = None, None
        if not should_compute:
            # Apply caching
            logger.info("Skipping forward pass through remaining blocks")
            hs = self.shared_state.tail_block_residuals[0] + output[hs_output_idx]
            if ehs_output_idx is not None:
                ehs = self.shared_state.tail_block_residuals[1] + output[ehs_output_idx]

            if isinstance(output, tuple):
                return_output = [None] * len(output)
                return_output[hs_output_idx] = hs
                return_output[ehs_output_idx] = ehs
                return_output = tuple(return_output)
            else:
                return_output = hs
            return return_output
        else:
            logger.info("Computing forward pass through remaining blocks")
            if isinstance(output, tuple):
                head_block_output = [None] * len(output)
                head_block_output[0] = output[hs_output_idx]
                head_block_output[1] = output[ehs_output_idx]
            else:
                head_block_output = output
            self.shared_state.head_block_output = head_block_output
            self.shared_state.head_block_residual = hs_residual
            return output

    def reset_state(self, module):
        self.shared_state.reset()
        return module

    def _should_compute_remaining_blocks(self, hs_residual: torch.Tensor) -> bool:
        if self.shared_state.head_block_residual is None:
            return True
        prev_hs_residual = self.shared_state.head_block_residual
        hs_absmean = (hs_residual - prev_hs_residual).abs().mean()
        prev_hs_mean = prev_hs_residual.abs().mean()
        diff = (hs_absmean / prev_hs_mean).item()
        logger.info(f"Diff: {diff}, Threshold: {self.threshold}")
        return diff > self.threshold


class FBCBlockHook(ModelHook):
    def __init__(self, shared_state: FBCSharedBlockState, is_tail: bool = False):
        super().__init__()
        self.shared_state = shared_state
        self.is_tail = is_tail

    def initialize_hook(self, module):
        inputs = inspect.signature(module.__class__.forward)
        inputs_index_to_str = dict(enumerate(inputs.parameters.keys()))
        inputs_str_to_index = {v: k for k, v in inputs_index_to_str.items()}

        try:
            outputs = _extract_return_information(module.__class__.forward)
            outputs_index_to_str = dict(enumerate(outputs))
            outputs_str_to_index = {v: k for k, v in outputs_index_to_str.items()}
        except RuntimeError:
            logger.error(f"Failed to extract return information for {module.__class__}")
            raise NotImplementedError(
                f"Module {module.__class__} is not supported with FirstBlockCache. Please open an issue at "
                f"https://github.com/huggingface/diffusers to notify us about the error with a minimal example "
                f"in order for us to add support for this module."
            )

        self._inputs_index_to_str = inputs_index_to_str
        self._inputs_str_to_index = inputs_str_to_index
        self._outputs_index_to_str = outputs_index_to_str
        self._outputs_str_to_index = outputs_str_to_index
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        hs_input_idx = self._inputs_str_to_index.get("hidden_states")
        ehs_input_idx = self._inputs_str_to_index.get("encoder_hidden_states", None)
        original_hs = kwargs.get("hidden_states", None)
        original_ehs = kwargs.get("encoder_hidden_states", None)
        original_hs = original_hs if original_hs is not None else args[hs_input_idx]
        if ehs_input_idx is not None:
            original_ehs = original_ehs if original_ehs is not None else args[ehs_input_idx]

        hs_output_idx = self._outputs_str_to_index.get("hidden_states")
        ehs_output_idx = self._outputs_str_to_index.get("encoder_hidden_states", None)
        assert (ehs_input_idx is None) == (ehs_output_idx is None)

        if self.shared_state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                hs_residual, ehs_residual = None, None
                if isinstance(output, tuple):
                    hs_residual = output[hs_output_idx] - self.shared_state.head_block_output[0]
                    ehs_residual = output[ehs_output_idx] - self.shared_state.head_block_output[1]
                else:
                    hs_residual = output - self.shared_state.head_block_output
                self.shared_state.tail_block_residuals = (hs_residual, ehs_residual)
            return output

        output_count = len(self._outputs_index_to_str.keys())
        return_output = [None] * output_count if output_count > 1 else original_hs
        if output_count == 1:
            return_output = original_hs
        else:
            return_output[hs_output_idx] = original_hs
            return_output[ehs_output_idx] = original_ehs
        return return_output


def apply_first_block_cache(module: torch.nn.Module, config: FirstBlockCacheConfig) -> None:
    shared_state = FBCSharedBlockState()
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for block in submodule:
            remaining_blocks.append((name, block))

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.debug(f"Apply FBCHeadBlockHook to '{head_block_name}'")
    apply_fbc_head_block_hook(head_block, shared_state, config.threshold)

    for name, block in remaining_blocks:
        logger.debug(f"Apply FBCBlockHook to '{name}'")
        apply_fbc_block_hook(block, shared_state)

    logger.debug(f"Apply FBCBlockHook to tail block '{tail_block_name}'")
    apply_fbc_block_hook(tail_block, shared_state, is_tail=True)


def apply_fbc_head_block_hook(block: torch.nn.Module, state: FBCSharedBlockState, threshold: float) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCHeadBlockHook(state, threshold)
    registry.register_hook(hook, _FBC_LEADER_BLOCK_HOOK)


def apply_fbc_block_hook(block: torch.nn.Module, state: FBCSharedBlockState, is_tail: bool = False) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    hook = FBCBlockHook(state, is_tail)
    registry.register_hook(hook, _FBC_BLOCK_HOOK)
