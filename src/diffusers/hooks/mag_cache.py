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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_MAG_CACHE_LEADER_BLOCK_HOOK = "mag_cache_leader_block_hook"
_MAG_CACHE_BLOCK_HOOK = "mag_cache_block_hook"

# Default Mag Ratios for Flux models (Dev/Schnell) are provided for convenience.
# Users must explicitly pass these to the config if using Flux.
# Reference: https://github.com/Zehong-Ma/MagCache
FLUX_MAG_RATIOS = torch.tensor(
    [1.0]
    + [
        1.21094,
        1.11719,
        1.07812,
        1.0625,
        1.03906,
        1.03125,
        1.03906,
        1.02344,
        1.03125,
        1.02344,
        0.98047,
        1.01562,
        1.00781,
        1.0,
        1.00781,
        1.0,
        1.00781,
        1.0,
        1.0,
        0.99609,
        0.99609,
        0.98047,
        0.98828,
        0.96484,
        0.95703,
        0.93359,
        0.89062,
    ]
)


def nearest_interp(src_array: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Interpolate the source array to the target length using nearest neighbor interpolation.
    """
    src_length = len(src_array)
    if target_length == 1:
        return src_array[-1:]

    scale = (src_length - 1) / (target_length - 1)
    grid = torch.arange(target_length, device=src_array.device, dtype=torch.float32)
    mapped_indices = torch.round(grid * scale).long()
    return src_array[mapped_indices]


@dataclass
class MagCacheConfig:
    r"""
    Configuration for [MagCache](https://github.com/Zehong-Ma/MagCache).

    Args:
        threshold (`float`, defaults to `0.06`):
            The threshold for the accumulated error. If the accumulated error is below this threshold, the block
            computation is skipped. A higher threshold allows for more aggressive skipping (faster) but may degrade
            quality.
        max_skip_steps (`int`, defaults to `3`):
            The maximum number of consecutive steps that can be skipped (K in the paper).
        retention_ratio (`float`, defaults to `0.2`):
            The fraction of initial steps during which skipping is disabled to ensure stability. For example, if
            `num_inference_steps` is 28 and `retention_ratio` is 0.2, the first 6 steps will never be skipped.
        num_inference_steps (`int`, defaults to `28`):
            The number of inference steps used in the pipeline. This is required to interpolate `mag_ratios` correctly.
        mag_ratios (`torch.Tensor`, *optional*):
            The pre-computed magnitude ratios for the model. These are checkpoint-dependent. If not provided, you must
            set `calibrate=True` to calculate them for your specific model. For Flux models, you can use
            `diffusers.hooks.mag_cache.FLUX_MAG_RATIOS`.
        calibrate (`bool`, defaults to `False`):
            If True, enables calibration mode. In this mode, no blocks are skipped. Instead, the hook calculates the
            magnitude ratios for the current run and logs them at the end. Use this to obtain `mag_ratios` for new
            models or schedulers.
    """

    threshold: float = 0.06
    max_skip_steps: int = 3
    retention_ratio: float = 0.2
    num_inference_steps: int = 28
    mag_ratios: Optional[Union[torch.Tensor, List[float]]] = None
    calibrate: bool = False

    def __post_init__(self):
        # User MUST provide ratios OR enable calibration.
        if self.mag_ratios is None and not self.calibrate:
            raise ValueError(
                " `mag_ratios` must be provided for MagCache inference because these ratios are model-dependent.\n"
                "To get them for your model:\n"
                "1. Initialize `MagCacheConfig(calibrate=True, ...)`\n"
                "2. Run inference on your model once.\n"
                "3. Copy the printed ratios array and pass it to `mag_ratios` in the config.\n"
                "For Flux models, you can import `FLUX_MAG_RATIOS` from `diffusers.hooks.mag_cache`."
            )

        if not self.calibrate and self.mag_ratios is not None:
            if not torch.is_tensor(self.mag_ratios):
                self.mag_ratios = torch.tensor(self.mag_ratios)

            if len(self.mag_ratios) != self.num_inference_steps:
                logger.debug(
                    f"Interpolating mag_ratios from length {len(self.mag_ratios)} to {self.num_inference_steps}"
                )
                self.mag_ratios = nearest_interp(self.mag_ratios, self.num_inference_steps)


class MagCacheState(BaseState):
    def __init__(self) -> None:
        super().__init__()
        # Cache for the residual (output - input) from the *previous* timestep
        self.previous_residual: torch.Tensor = None

        # State inputs/outputs for the current forward pass
        self.head_block_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
        self.should_compute: bool = True

        # MagCache accumulators
        self.accumulated_ratio: float = 1.0
        self.accumulated_err: float = 0.0
        self.accumulated_steps: int = 0

        # Current step counter (timestep index)
        self.step_index: int = 0

        # Calibration storage
        self.calibration_ratios: List[float] = []

    def reset(self):
        self.previous_residual = None
        self.should_compute = True
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.step_index = 0
        self.calibration_ratios = []


class MagCacheHeadHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager, config: MagCacheConfig):
        self.state_manager = state_manager
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")

        arg_name = self._metadata.hidden_states_argument_name
        hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

        state: MagCacheState = self.state_manager.get_state()
        state.head_block_input = hidden_states

        should_compute = True

        if self.config.calibrate:
            # Never skip during calibration
            should_compute = True
        else:
            # MagCache Logic
            current_step = state.step_index
            if current_step >= len(self.config.mag_ratios):
                current_scale = 1.0
            else:
                current_scale = self.config.mag_ratios[current_step]

            retention_step = int(self.config.retention_ratio * self.config.num_inference_steps + 0.5)

            if current_step >= retention_step:
                state.accumulated_ratio *= current_scale
                state.accumulated_steps += 1
                state.accumulated_err += abs(1.0 - state.accumulated_ratio)

                if (
                    state.previous_residual is not None
                    and state.accumulated_err <= self.config.threshold
                    and state.accumulated_steps <= self.config.max_skip_steps
                ):
                    should_compute = False
                else:
                    state.accumulated_ratio = 1.0
                    state.accumulated_steps = 0
                    state.accumulated_err = 0.0

        state.should_compute = should_compute

        if not should_compute:
            logger.debug(f"MagCache: Skipping step {state.step_index}")
            # Apply MagCache: Output = Input + Previous Residual

            output = hidden_states
            res = state.previous_residual

            if res.device != output.device:
                res = res.to(output.device)

            # Attempt to apply residual handling shape mismatches (e.g., text+image vs image only)
            if res.shape == output.shape:
                output = output + res
            elif (
                output.ndim == 3
                and res.ndim == 3
                and output.shape[0] == res.shape[0]
                and output.shape[2] == res.shape[2]
            ):
                # Assuming concatenation where image part is at the end (standard in Flux/SD3)
                diff = output.shape[1] - res.shape[1]
                if diff > 0:
                    output = output.clone()
                    output[:, diff:, :] = output[:, diff:, :] + res
                else:
                    logger.warning(
                        f"MagCache: Dimension mismatch. Input {output.shape}, Residual {res.shape}. "
                        "Cannot apply residual safely. Returning input without residual."
                    )
            else:
                logger.warning(
                    f"MagCache: Dimension mismatch. Input {output.shape}, Residual {res.shape}. "
                    "Cannot apply residual safely. Returning input without residual."
                )

            if self._metadata.return_encoder_hidden_states_index is not None:
                original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = output
                ret_list[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
                return tuple(ret_list)
            else:
                return output

        else:
            # Compute original forward
            output = self.fn_ref.original_forward(*args, **kwargs)
            return output

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class MagCacheBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False, config: MagCacheConfig = None):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self.config = config
        self._metadata = None

    def initialize_hook(self, module):
        unwrapped_module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(unwrapped_module.__class__)
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("inference")
        state: MagCacheState = self.state_manager.get_state()

        if not state.should_compute:
            arg_name = self._metadata.hidden_states_argument_name
            hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

            if self.is_tail:
                # Still need to advance step index even if we skip
                self._advance_step(state)

            if self._metadata.return_encoder_hidden_states_index is not None:
                encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = hidden_states
                ret_list[self._metadata.return_encoder_hidden_states_index] = encoder_hidden_states
                return tuple(ret_list)

            return hidden_states

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_tail:
            # Calculate residual for next steps
            if isinstance(output, tuple):
                out_hidden = output[self._metadata.return_hidden_states_index]
            else:
                out_hidden = output

            in_hidden = state.head_block_input

            if in_hidden is None:
                return output

            # Determine residual
            if out_hidden.shape == in_hidden.shape:
                residual = out_hidden - in_hidden
            elif out_hidden.ndim == 3 and in_hidden.ndim == 3 and out_hidden.shape[2] == in_hidden.shape[2]:
                diff = in_hidden.shape[1] - out_hidden.shape[1]
                if diff == 0:
                    residual = out_hidden - in_hidden
                else:
                    residual = out_hidden - in_hidden  # Fallback to matching tail
            else:
                # Fallback for completely mismatched shapes
                residual = out_hidden

            if self.config.calibrate:
                self._perform_calibration_step(state, residual)

            state.previous_residual = residual
            self._advance_step(state)

        return output

    def _perform_calibration_step(self, state: MagCacheState, current_residual: torch.Tensor):
        if state.previous_residual is None:
            # First step has no previous residual to compare against.
            # log 1.0 as a neutral starting point.
            ratio = 1.0
        else:
            # MagCache Calibration Formula: mean(norm(curr) / norm(prev))
            # norm(dim=-1) gives magnitude of each token vector
            curr_norm = torch.linalg.norm(current_residual.float(), dim=-1)
            prev_norm = torch.linalg.norm(state.previous_residual.float(), dim=-1)

            # Avoid division by zero
            ratio = (curr_norm / (prev_norm + 1e-8)).mean().item()

        state.calibration_ratios.append(ratio)

    def _advance_step(self, state: MagCacheState):
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            # End of inference loop
            if self.config.calibrate:
                print("\n[MagCache] Calibration Complete. Copy these values to MagCacheConfig(mag_ratios=...):")
                print(f"{state.calibration_ratios}\n")
                logger.info(f"MagCache Calibration Results: {state.calibration_ratios}")

            # Reset state
            state.step_index = 0
            state.accumulated_ratio = 1.0
            state.accumulated_steps = 0
            state.accumulated_err = 0.0
            state.previous_residual = None
            state.calibration_ratios = []


def apply_mag_cache(module: torch.nn.Module, config: MagCacheConfig) -> None:
    """
    Applies MagCache to a given module (typically a Transformer).

    Args:
        module (`torch.nn.Module`):
            The module to apply MagCache to.
        config (`MagCacheConfig`):
            The configuration for MagCache.
    """
    # Initialize registry on the root module so the Pipeline can set context.
    HookRegistry.check_if_exists_or_initialize(module)

    state_manager = StateManager(MagCacheState, (), {})
    remaining_blocks = []

    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    if not remaining_blocks:
        logger.warning("MagCache: No transformer blocks found to apply hooks.")
        return

    # Handle single-block models
    if len(remaining_blocks) == 1:
        name, block = remaining_blocks[0]
        logger.info(f"MagCache: Applying Head+Tail Hooks to single block '{name}'")
        _apply_mag_cache_block_hook(block, state_manager, config, is_tail=True)
        _apply_mag_cache_head_hook(block, state_manager, config)
        return

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.info(f"MagCache: Applying Head Hook to {head_block_name}")
    _apply_mag_cache_head_hook(head_block, state_manager, config)

    for name, block in remaining_blocks:
        _apply_mag_cache_block_hook(block, state_manager, config)

    logger.info(f"MagCache: Applying Tail Hook to {tail_block_name}")
    _apply_mag_cache_block_hook(tail_block, state_manager, config, is_tail=True)


def _apply_mag_cache_head_hook(block: torch.nn.Module, state_manager: StateManager, config: MagCacheConfig) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    # Automatically remove existing hook to allow re-application (e.g. switching modes)
    if registry.get_hook(_MAG_CACHE_LEADER_BLOCK_HOOK) is not None:
        registry.remove_hook(_MAG_CACHE_LEADER_BLOCK_HOOK)

    hook = MagCacheHeadHook(state_manager, config)
    registry.register_hook(hook, _MAG_CACHE_LEADER_BLOCK_HOOK)


def _apply_mag_cache_block_hook(
    block: torch.nn.Module,
    state_manager: StateManager,
    config: MagCacheConfig,
    is_tail: bool = False,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    # Automatically remove existing hook to allow re-application
    if registry.get_hook(_MAG_CACHE_BLOCK_HOOK) is not None:
        registry.remove_hook(_MAG_CACHE_BLOCK_HOOK)

    hook = MagCacheBlockHook(state_manager, is_tail, config)
    registry.register_hook(hook, _MAG_CACHE_BLOCK_HOOK)
