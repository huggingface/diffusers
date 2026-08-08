# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..utils import get_logger
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = get_logger(__name__)  # pylint: disable=invalid-name

_RESILPHASE_LEADER_BLOCK_HOOK = "resilphase_leader_block_hook"
_RESILPHASE_BLOCK_HOOK = "resilphase_block_hook"
_RESILPHASE_DENOISER_HOOK = "resilphase_denoiser_hook"
_CONTROL_RESIDUAL_ARGUMENTS = ("controlnet_block_samples", "controlnet_single_block_samples")


@dataclass
class ResilPhaseCacheConfig:
    r"""
    Configuration for ResilPhase cache.

    ResilPhase approximates the residual produced by a denoiser's transformer block stack with barycentric Lagrange
    interpolation on a normalized phase axis. On prediction steps, the expensive transformer blocks are skipped and the
    predicted residual is added to the block stack input.

    Args:
        cache_interval (`int`, defaults to `6`):
            Number of denoising steps between full transformer block computations. The intermediate `cache_interval -
            1` steps use ResilPhase predictions.
        warmup_steps (`int`, defaults to `3`):
            Number of initial denoising steps that always perform full computations to initialize the interpolation
            history.
        max_order (`int`, defaults to `1`):
            Maximum interpolation order. ResilPhase retains at most `max_order + 1` fully computed residuals.
        mapping_method (`str`, defaults to `"balanced"`):
            Phase-axis mapping. Must be `"balanced"` for a hyperbolic-tangent mapping or `"chebyshev"` for Chebyshev
            nodes.
        balance_alpha (`float`, defaults to `0.55`):
            Scale of the hyperbolic-tangent phase mapping. Only used when `mapping_method="balanced"`.
    """

    cache_interval: int = 6
    warmup_steps: int = 3
    max_order: int = 1
    mapping_method: Literal["balanced", "chebyshev"] = "balanced"
    balance_alpha: float = 0.55

    def __post_init__(self) -> None:
        if self.cache_interval < 1:
            raise ValueError("`cache_interval` must be greater than zero.")
        if self.warmup_steps < 0:
            raise ValueError("`warmup_steps` must be greater than or equal to zero.")
        if self.max_order < 0:
            raise ValueError("`max_order` must be greater than or equal to zero.")
        if self.mapping_method not in {"balanced", "chebyshev"}:
            raise ValueError('`mapping_method` must be either "balanced" or "chebyshev".')
        if self.balance_alpha <= 0:
            raise ValueError("`balance_alpha` must be greater than zero.")


class ResilPhaseState(BaseState):
    def __init__(self, config: ResilPhaseCacheConfig) -> None:
        self.config = config
        self.step_index = -1
        self.skipped_steps = 0
        self.should_compute = True
        self.bypass = False

        self.stack_input: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self.history_steps: list[int] = []
        self.history_residuals: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def reset(self) -> None:
        self.step_index = -1
        self.skipped_steps = 0
        self.should_compute = True
        self.bypass = False
        self.stack_input = None
        self.history_steps = []
        self.history_residuals = []

    def start_step(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None) -> None:
        self.step_index += 1
        has_history = len(self.history_residuals) > 0
        is_warmup = self.step_index < self.config.warmup_steps
        reached_refresh = self.skipped_steps >= self.config.cache_interval - 1
        self.should_compute = is_warmup or not has_history or reached_refresh

        if self.should_compute:
            self.skipped_steps = 0
            self.stack_input = (hidden_states, encoder_hidden_states)
        else:
            self.skipped_steps += 1
            self.stack_input = None

    def update(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None) -> None:
        if self.stack_input is None:
            raise ValueError("Cannot update ResilPhase state without a transformer block stack input.")

        input_hidden_states, input_encoder_hidden_states = self.stack_input
        hidden_states_residual = (hidden_states - input_hidden_states).detach().clone()
        encoder_hidden_states_residual = None
        if encoder_hidden_states is not None:
            encoder_hidden_states_residual = (encoder_hidden_states - input_encoder_hidden_states).detach().clone()

        self.history_steps.append(self.step_index)
        self.history_residuals.append((hidden_states_residual, encoder_hidden_states_residual))
        history_size = self.config.max_order + 1
        self.history_steps = self.history_steps[-history_size:]
        self.history_residuals = self.history_residuals[-history_size:]
        self.stack_input = None

    @torch.compiler.disable
    def predict(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if len(self.history_residuals) == 1:
            return self.history_residuals[0]

        if self.config.mapping_method == "chebyshev":
            num_nodes = len(self.history_steps)
            phase_nodes = [math.cos((2 * index + 1) * math.pi / (2 * num_nodes)) for index in range(num_nodes)]
            phase_nodes.sort(reverse=True)

            left_step, right_step = self.history_steps[-2:]
            left_node, right_node = phase_nodes[-2:]
            slope = (right_node - left_node) / (right_step - left_step)
            target_phase = right_node + slope * (self.step_index - right_step)
            target_phase = min(1.0, target_phase)
        else:
            mean_step = sum(self.history_steps) / len(self.history_steps)
            max_distance = max(abs(step - mean_step) for step in self.history_steps)
            phase_nodes = [
                math.tanh(self.config.balance_alpha * (step - mean_step) / max_distance) for step in self.history_steps
            ]
            target_phase = math.tanh(self.config.balance_alpha * (self.step_index - mean_step) / max_distance)

        barycentric_weights = []
        for index, node in enumerate(phase_nodes):
            log_weight = 0.0
            sign = 1
            for other_index, other_node in enumerate(phase_nodes):
                if index != other_index:
                    difference = node - other_node
                    log_weight -= math.log(abs(difference))
                    if difference < 0:
                        sign *= -1
            log_weight = max(-700, min(700, log_weight))
            barycentric_weights.append(sign * math.exp(log_weight))

        total_absolute_weight = sum(abs(weight) for weight in barycentric_weights)
        barycentric_weights = [
            weight / total_absolute_weight * len(barycentric_weights) for weight in barycentric_weights
        ]

        hidden_states_numerator = None
        encoder_hidden_states_numerator = None
        denominator = 0.0
        for index, (hidden_states_residual, encoder_hidden_states_residual) in enumerate(self.history_residuals):
            distance = target_phase - phase_nodes[index]
            if abs(distance) < 1e-12:
                return hidden_states_residual, encoder_hidden_states_residual

            coefficient = barycentric_weights[index] / distance
            hidden_states_term = coefficient * hidden_states_residual
            hidden_states_numerator = (
                hidden_states_term if hidden_states_numerator is None else hidden_states_numerator + hidden_states_term
            )
            if encoder_hidden_states_residual is not None:
                encoder_hidden_states_term = coefficient * encoder_hidden_states_residual
                encoder_hidden_states_numerator = (
                    encoder_hidden_states_term
                    if encoder_hidden_states_numerator is None
                    else encoder_hidden_states_numerator + encoder_hidden_states_term
                )
            denominator += coefficient

        hidden_states_prediction = hidden_states_numerator / denominator
        encoder_hidden_states_prediction = (
            None if encoder_hidden_states_numerator is None else encoder_hidden_states_numerator / denominator
        )
        return hidden_states_prediction, encoder_hidden_states_prediction


def _get_block_inputs(
    metadata: TransformerBlockMetadata, args: tuple, kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
    encoder_hidden_states = None
    if metadata.return_encoder_hidden_states_index is not None:
        encoder_hidden_states = metadata._get_parameter_from_args_kwargs("encoder_hidden_states", args, kwargs)
    return hidden_states, encoder_hidden_states


def _get_block_outputs(
    metadata: TransformerBlockMetadata, output: torch.Tensor | tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        hidden_states = output[metadata.return_hidden_states_index]
        encoder_hidden_states = (
            None
            if metadata.return_encoder_hidden_states_index is None
            else output[metadata.return_encoder_hidden_states_index]
        )
        return hidden_states, encoder_hidden_states
    return output, None


def _pack_block_outputs(
    metadata: TransformerBlockMetadata,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if metadata.return_encoder_hidden_states_index is None:
        return hidden_states

    output = [None, None]
    output[metadata.return_hidden_states_index] = hidden_states
    output[metadata.return_encoder_hidden_states_index] = encoder_hidden_states
    return tuple(output)


class ResilPhaseHeadBlockHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._metadata = None

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: ResilPhaseState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        state.start_step(hidden_states, encoder_hidden_states)

        if state.should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        hidden_states_residual, encoder_hidden_states_residual = state.predict()
        hidden_states = hidden_states + hidden_states_residual
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_residual
        return _pack_block_outputs(self._metadata, hidden_states, encoder_hidden_states)


class ResilPhaseBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, is_tail: bool = False) -> None:
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self._metadata = None

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: ResilPhaseState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        if state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                hidden_states, encoder_hidden_states = _get_block_outputs(self._metadata, output)
                state.update(hidden_states, encoder_hidden_states)
            return output

        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        return _pack_block_outputs(self._metadata, hidden_states, encoder_hidden_states)


class ResilPhaseDenoiserHook(ModelHook):
    def __init__(self, state_manager: StateManager) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._control_argument_indices = {}

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        self._control_argument_indices = {
            name: parameters.index(name) for name in _CONTROL_RESIDUAL_ARGUMENTS if name in parameters
        }
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: ResilPhaseState = self.state_manager.get_state()
        state.bypass = any(kwargs.get(name) is not None for name in _CONTROL_RESIDUAL_ARGUMENTS)
        if not state.bypass:
            state.bypass = any(
                index < len(args) and args[index] is not None for index in self._control_argument_indices.values()
            )

        try:
            return self.fn_ref.original_forward(*args, **kwargs)
        finally:
            state.bypass = False


def apply_resilphase_cache(module: torch.nn.Module, config: ResilPhaseCacheConfig) -> None:
    r"""Apply ResilPhase cache to the transformer blocks of a denoiser.

    Args:
        module (`torch.nn.Module`):
            The denoiser module whose transformer block stack should be cached.
        config (`ResilPhaseCacheConfig`):
            Configuration for ResilPhase cache.

    Example:
        ```python
        >>> from diffusers import FluxPipeline, ResilPhaseCacheConfig

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        >>> pipe.transformer.enable_cache(ResilPhaseCacheConfig())
        ```
    """

    transformer_blocks = []
    for name, submodule in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            transformer_blocks.append((f"{name}.{index}", block))

    if len(transformer_blocks) < 2:
        raise ValueError("ResilPhase cache requires a denoiser with at least two transformer blocks.")

    state_manager = StateManager(ResilPhaseState, init_args=(config,))
    head_block_name, head_block = transformer_blocks.pop(0)
    tail_block_name, tail_block = transformer_blocks.pop(-1)

    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(ResilPhaseDenoiserHook(state_manager), _RESILPHASE_DENOISER_HOOK)

    logger.debug(f"Applying ResilPhaseHeadBlockHook to '{head_block_name}'")
    registry = HookRegistry.check_if_exists_or_initialize(head_block)
    registry.register_hook(ResilPhaseHeadBlockHook(state_manager), _RESILPHASE_LEADER_BLOCK_HOOK)

    for name, block in transformer_blocks:
        logger.debug(f"Applying ResilPhaseBlockHook to '{name}'")
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(ResilPhaseBlockHook(state_manager), _RESILPHASE_BLOCK_HOOK)

    logger.debug(f"Applying ResilPhaseBlockHook to tail block '{tail_block_name}'")
    registry = HookRegistry.check_if_exists_or_initialize(tail_block)
    registry.register_hook(ResilPhaseBlockHook(state_manager, is_tail=True), _RESILPHASE_BLOCK_HOOK)
