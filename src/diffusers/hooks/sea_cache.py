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
from typing import Any, Callable, Literal

import torch

from ..utils import logging
from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_SEA_CACHE_ROOT_HOOK = "sea_cache_root"
_SEA_CACHE_LEADER_BLOCK_HOOK = "sea_cache_leader_block"
_SEA_CACHE_BLOCK_HOOK = "sea_cache_block"
_SEA_CACHE_POST_NORM_HOOK = "sea_cache_post_norm"


@dataclass
class SeaCacheConfig:
    r"""
    Configuration for [SeaCache](https://huggingface.co/papers/2602.18993).

    SeaCache is disabled by default and only activates after this configuration is passed to
    `transformer.enable_cache(config)`.

    SeaCache compares Spectral-Evolution-Aware (SEA) indicators between scheduler steps. If their accumulated relative
    change stays below `threshold`, the expensive language-model hidden transform is replaced with a cached residual.
    For Cosmos 3, the residual spans the decoder stack and final pathway normalization; input packing and modality
    prediction heads still execute.

    Args:
        threshold (`float`, defaults to `0.25`):
            Accumulated relative-L1 budget. Larger values reuse the cache more often.
        residual_order (`int`, defaults to `1`):
            Order used to predict the generation-stream language-model residual. `0` directly reuses the most recent
            residual and `1` linearly extrapolates from the two most recent full executions.
        retention_steps (`int`, defaults to `1`):
            Number of initial scheduler steps that always execute in full.
        cache_end_steps (`int`, defaults to `1`):
            Number of final scheduler steps that always execute in full.
        max_consecutive_cached (`int`, defaults to `2`):
            Maximum consecutive residual reuses per cache context before forcing a full execution. `0` disables the
            limit.
        power_exp (`float`, defaults to `3.0`):
            Exponent of the SEA clean-signal power prior. SeaCache uses `3.0` for video features.
        current_step_callback (`Callable[[], int]`):
            Callback returning the current scheduler step index.
        current_sigma_callback (`Callable[[], float]`):
            Callback returning the exact current scheduler sigma in `[0, 1]`.
        num_inference_steps_callback (`Callable[[], int]`):
            Callback returning the number of scheduler steps in the current pipeline call.
        raw_vision_callback (`Callable`, *optional*):
            Advanced model adapter returning raw vision latents with shape `(C, T, H, W)`. Cosmos 3 uses its native
            adapter when this is omitted.

    Example:
        ```python
        >>> from diffusers import Cosmos3OmniPipeline, SeaCacheConfig

        >>> pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano")
        >>> pipe.transformer.enable_cache(
        ...     SeaCacheConfig(
        ...         current_step_callback=lambda: pipe.current_step_index,
        ...         current_sigma_callback=lambda: pipe.current_sigma,
        ...         num_inference_steps_callback=lambda: pipe.num_timesteps,
        ...     )
        ... )
        ```
    """

    threshold: float = 0.25
    residual_order: int = 1
    retention_steps: int = 1
    cache_end_steps: int = 1
    max_consecutive_cached: int = 2
    power_exp: float = 3.0
    current_step_callback: Callable[[], int] = None
    current_sigma_callback: Callable[[], float] = None
    num_inference_steps_callback: Callable[[], int] = None
    raw_vision_callback: Callable[
        [torch.nn.Module, tuple[Any, ...], dict[str, Any]],
        list[torch.Tensor] | None,
    ] = None

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError(f"`threshold` must be non-negative, got {self.threshold}.")
        if self.residual_order not in (0, 1):
            raise ValueError(f"`residual_order` must be 0 or 1, got {self.residual_order}.")
        if self.retention_steps < 0:
            raise ValueError(f"`retention_steps` must be non-negative, got {self.retention_steps}.")
        if self.cache_end_steps < 0:
            raise ValueError(f"`cache_end_steps` must be non-negative, got {self.cache_end_steps}.")
        if (
            isinstance(self.max_consecutive_cached, bool)
            or not isinstance(self.max_consecutive_cached, int)
            or self.max_consecutive_cached < 0
        ):
            raise ValueError(
                f"`max_consecutive_cached` must be a non-negative integer, got {self.max_consecutive_cached!r}."
            )
        if not math.isfinite(self.power_exp) or self.power_exp <= 0:
            raise ValueError(f"`power_exp` must be positive, got {self.power_exp}.")
        for name in (
            "current_step_callback",
            "current_sigma_callback",
            "num_inference_steps_callback",
            "raw_vision_callback",
        ):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"`{name}` must be callable or `None`.")


@dataclass
class _SeaCacheForwardMetadata:
    step_index: int
    sigma: float
    num_inference_steps: int
    raw_vision: list[torch.Tensor] | None = None


class SeaCacheContextState(BaseState):
    def __init__(self):
        self.history: list[tuple[int, torch.Tensor | None, torch.Tensor]] = []
        self.gate_key: tuple[int, float] | None = None
        self.gate_should_compute = True
        self.previous_indicator: list[torch.Tensor] | None = None
        self.accumulated_distance = 0.0
        self.consecutive_cached = 0
        self.skip_remaining = False
        self.full_execution_pending = False
        self.cacheable_execution = False
        self.step_index: int | None = None
        self.gen_input: torch.Tensor | None = None
        self.und_output: torch.Tensor | None = None
        self.cached_und_output: torch.Tensor | None = None
        self.cached_gen_residual: torch.Tensor | None = None
        self.gen_output: torch.Tensor | None = None

    def reset_forward(self):
        self.skip_remaining = False
        self.full_execution_pending = False
        self.cacheable_execution = False
        self.step_index = None
        self.gen_input = None
        self.und_output = None
        self.cached_und_output = None
        self.cached_gen_residual = None
        self.gen_output = None

    def reset_trajectory(self):
        self.history = []
        self.gate_key = None
        self.gate_should_compute = True
        self.previous_indicator = None
        self.accumulated_distance = 0.0
        self.consecutive_cached = 0

    def reset(self):
        self.reset_trajectory()
        self.reset_forward()


class SeaCacheSharedState:
    def __init__(self):
        self._warned_messages: set[str] = set()
        self.reset()

    def reset(self):
        self.forward_metadata: _SeaCacheForwardMetadata | None = None

    def warn_once(self, message: str):
        if message not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(message)

    def mark_fail_open(self, message: str):
        self.warn_once(message)

    def resolve_gate(
        self,
        state: SeaCacheContextState,
        metadata: _SeaCacheForwardMetadata,
        indicator: list[torch.Tensor] | None,
        config: SeaCacheConfig,
    ) -> bool:
        gate_key = (metadata.step_index, metadata.sigma)
        if state.gate_key == gate_key:
            return state.gate_should_compute

        is_non_adjacent = state.gate_key is not None and metadata.step_index != state.gate_key[0] + 1
        if is_non_adjacent:
            self.mark_fail_open("SeaCache received non-adjacent scheduler steps; running full.")
            state.reset_trajectory()
        is_retained = metadata.step_index < config.retention_steps
        is_in_cache_end = metadata.step_index >= metadata.num_inference_steps - config.cache_end_steps
        is_first_observation = state.previous_indicator is None
        is_max_consecutive = bool(
            config.max_consecutive_cached and state.consecutive_cached >= config.max_consecutive_cached
        )
        invalid_gate = is_non_adjacent or indicator is None
        forced_compute = invalid_gate or is_retained or is_in_cache_end or is_first_observation or is_max_consecutive
        candidate_accumulated_distance = 0.0

        if forced_compute:
            natural_should_compute = True
        else:
            if len(indicator) != len(state.previous_indicator) or not indicator:
                distance = float("inf")
                invalid_gate = True
            else:
                distance = 0.0
                for current, previous in zip(indicator, state.previous_indicator):
                    if (
                        current.shape != previous.shape
                        or current.device != previous.device
                        or current.dtype != previous.dtype
                    ):
                        distance = float("inf")
                        invalid_gate = True
                        break
                    numerator = (current.float() - previous.float()).abs().mean()
                    denominator = previous.float().abs().mean() + 1e-16
                    distance += float((numerator / denominator).detach().cpu())
                distance /= len(indicator)

            if not math.isfinite(distance):
                invalid_gate = True
                self.mark_fail_open("SeaCache indicator history changed shape, device, or dtype; running full.")
            candidate_accumulated_distance = state.accumulated_distance + distance
            natural_should_compute = invalid_gate or candidate_accumulated_distance >= config.threshold

        should_compute = natural_should_compute

        if is_max_consecutive:
            should_compute = True

        state.accumulated_distance = 0.0 if should_compute else candidate_accumulated_distance

        state.gate_key = gate_key
        state.gate_should_compute = should_compute
        state.previous_indicator = None if indicator is None else [value.detach() for value in indicator]
        return should_compute


def _get_block_inputs(
    metadata: TransformerBlockMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = metadata._get_parameter_from_args_kwargs(metadata.hidden_states_argument_name, args, kwargs)
    encoder_hidden_states = None
    if metadata.return_encoder_hidden_states_index is not None:
        encoder_hidden_states = metadata._get_parameter_from_args_kwargs(
            metadata.encoder_hidden_states_argument_name, args, kwargs
        )
    return hidden_states, encoder_hidden_states


def _build_block_output(
    metadata: TransformerBlockMetadata,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if metadata.return_encoder_hidden_states_index is None:
        return hidden_states

    output = [None] * (max(metadata.return_hidden_states_index, metadata.return_encoder_hidden_states_index) + 1)
    output[metadata.return_hidden_states_index] = hidden_states
    output[metadata.return_encoder_hidden_states_index] = encoder_hidden_states
    return tuple(output)


def _get_block_outputs(
    metadata: TransformerBlockMetadata, output: torch.Tensor | tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        hidden_states = output[metadata.return_hidden_states_index]
        encoder_hidden_states = (
            output[metadata.return_encoder_hidden_states_index]
            if metadata.return_encoder_hidden_states_index is not None
            else None
        )
        return hidden_states, encoder_hidden_states
    return output, None


def _record_full_execution(
    config: SeaCacheConfig,
    state: SeaCacheContextState,
    gen_output: torch.Tensor,
    und_output: torch.Tensor | None,
    requires_und_output: bool = True,
) -> None:
    state.consecutive_cached = 0
    if (
        state.cacheable_execution
        and state.step_index is not None
        and state.gen_input is not None
        and (und_output is not None or not requires_und_output)
        and gen_output.shape == state.gen_input.shape
    ):
        state.history.append(
            (
                state.step_index,
                und_output.detach().clone() if und_output is not None else None,
                (gen_output - state.gen_input).detach().clone(),
            )
        )
        state.history = state.history[-(config.residual_order + 1) :]
    state.reset_forward()


def _prepare_cosmos3_raw_vision_metadata(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[torch.Tensor] | None:
    module = unwrap_module(module)
    bound_arguments = inspect.signature(module.__class__.forward).bind_partial(module, *args, **kwargs).arguments
    vision_tokens = bound_arguments.get("vision_tokens")
    vision_noisy_frame_indexes = bound_arguments.get("vision_noisy_frame_indexes")

    if (
        not isinstance(vision_tokens, (list, tuple))
        or not isinstance(vision_noisy_frame_indexes, (list, tuple))
        or len(vision_tokens) != len(vision_noisy_frame_indexes)
    ):
        return None

    raw_vision = []
    has_noisy_vision = False
    for latent, noisy_frame_indexes in zip(vision_tokens, vision_noisy_frame_indexes):
        if not isinstance(latent, torch.Tensor) or not isinstance(noisy_frame_indexes, torch.Tensor):
            return None
        if latent.ndim == 5:
            if latent.shape[0] != 1:
                return None
            latent = latent.squeeze(0)
        if latent.ndim != 4:
            return None

        noisy_frame_indexes = noisy_frame_indexes.flatten().to(device=latent.device, dtype=torch.long)
        if torch.any(noisy_frame_indexes < 0) or torch.any(noisy_frame_indexes >= latent.shape[1]):
            return None
        has_noisy_vision = has_noisy_vision or noisy_frame_indexes.numel() > 0
        raw_vision.append(latent)

    return raw_vision if raw_vision and has_noisy_vision else None


def _apply_sea_filter(
    hidden_states: torch.Tensor,
    sigma: float,
    power_exp: float,
) -> torch.Tensor:
    hidden_states_dtype = hidden_states.dtype
    hidden_states = hidden_states.contiguous().float()
    dimensions = (0, 1, 2)
    spectrum = torch.fft.fftn(hidden_states, dim=dimensions)

    sigma = max(1e-6, min(1.0 - 1e-6, sigma))
    signal_scale = 1.0 - sigma
    noise_scale = sigma
    gain = None
    for axis in dimensions:
        frequencies = torch.fft.fftfreq(hidden_states.shape[axis], device=hidden_states.device, dtype=torch.float32)
        clean_power = 1.0 / (frequencies.abs().pow(power_exp) + 1e-16)
        axis_gain = signal_scale * clean_power / (signal_scale**2 * clean_power + noise_scale**2 + 1e-16)
        axis_shape = [1] * hidden_states.ndim
        axis_shape[axis] = axis_gain.shape[0]
        gain = axis_gain.reshape(axis_shape) if gain is None else gain * axis_gain.reshape(axis_shape)

    # SeaCache Eq. (7): density-normalize the combined spatiotemporal response
    # to unit mean so cache distances are comparable across scheduler steps.
    mean_gain = gain.mean()
    if torch.isfinite(mean_gain) and mean_gain > 0:
        gain = gain / mean_gain
    return torch.fft.ifftn(spectrum * gain, dim=dimensions).real.to(hidden_states_dtype)


def _build_indicator(
    config: SeaCacheConfig,
    forward_metadata: _SeaCacheForwardMetadata,
) -> list[torch.Tensor] | None:
    if not forward_metadata.raw_vision:
        return None
    return [
        _apply_sea_filter(
            latent.movedim(0, -1),
            sigma=forward_metadata.sigma,
            power_exp=config.power_exp,
        ).detach()
        for latent in forward_metadata.raw_vision
    ]


def _parameter_sharding_types(module: torch.nn.Module) -> tuple[bool, bool]:
    """Return whether a module contains FSDP-managed and DTensor parameters."""
    has_fsdp = False
    has_dtensor = False
    for submodule in unwrap_module(module).modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            has_fsdp = True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            has_fsdp = True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if parameter_type.__name__ == "FlatParameter" and parameter_type.__module__.startswith(
                "torch.distributed.fsdp"
            ):
                has_fsdp = True
            elif parameter_type.__name__ == "DTensor" and parameter_type.__module__.startswith(
                "torch.distributed.tensor"
            ):
                has_dtensor = True
    return has_fsdp, has_dtensor


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Whether a block is managed by a parameter-sharding runtime that SeaCache cannot safely bypass."""
    return any(_parameter_sharding_types(module))


def _synchronize_compute_decision(should_compute: bool, device: torch.device) -> tuple[bool, bool]:
    """Synchronize a cache decision across the active distributed world.

    The two votes distinguish unanimous full/skip decisions from disagreement. Disagreement always resolves to full
    execution and resets cache trajectories.
    """
    votes = torch.tensor(
        [int(should_compute), int(not should_compute)],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(votes, op=torch.distributed.ReduceOp.MAX)
    return bool(votes[0].item()), bool(votes[0].item() and votes[1].item())


class SeaCacheRootHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        config: SeaCacheConfig,
        state_manager: StateManager,
        shared_state: SeaCacheSharedState,
        raw_vision_callback: Callable,
        use_stack_boundary: bool = False,
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.shared_state = shared_state
        self.raw_vision_callback = raw_vision_callback
        self.use_stack_boundary = use_stack_boundary

    def initialize_hook(self, module: torch.nn.Module):
        if not self.use_stack_boundary:
            return module

        unwrapped_module = unwrap_module(module)
        if not hasattr(unwrapped_module, "layers") or not unwrapped_module.layers:
            raise ValueError("SeaCache requires Cosmos 3 to expose a non-empty decoder stack.")
        unwrapped_module._sea_cache_prepare_decoder_stack = self.prepare_decoder_stack
        unwrapped_module._sea_cache_record_decoder_stack = self.record_decoder_stack
        return module

    def deinitalize_hook(self, module: torch.nn.Module):
        if self.use_stack_boundary:
            unwrapped_module = unwrap_module(module)
            for name in ("_sea_cache_prepare_decoder_stack", "_sea_cache_record_decoder_stack"):
                if hasattr(unwrapped_module, name):
                    delattr(unwrapped_module, name)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        self.shared_state.forward_metadata = None
        if self.state_manager._current_context is not None:
            self.state_manager.get_state().reset_forward()
        if torch.is_grad_enabled():
            self.shared_state.mark_fail_open(
                "SeaCache is inference-only; calls with autograd enabled run in fail-open mode."
            )
            return args, kwargs
        if self.state_manager._current_context is None:
            self.shared_state.mark_fail_open(
                "SeaCache requires a cache context for each transformer call; running in fail-open mode."
            )
            return args, kwargs
        callbacks = (
            self.config.current_step_callback,
            self.config.current_sigma_callback,
            self.config.num_inference_steps_callback,
        )
        if any(callback is None for callback in callbacks):
            self.shared_state.mark_fail_open(
                "SeaCache is running in fail-open mode because scheduler step, sigma, and step-count callbacks are "
                "required."
            )
            return args, kwargs

        try:
            step_index = self.config.current_step_callback()
            sigma = self.config.current_sigma_callback()
            num_inference_steps = self.config.num_inference_steps_callback()
            if isinstance(step_index, torch.Tensor):
                step_index = step_index.item()
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.item()
            if isinstance(num_inference_steps, torch.Tensor):
                num_inference_steps = num_inference_steps.item()
            step_index = int(step_index)
            sigma = float(sigma)
            num_inference_steps = int(num_inference_steps)
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            self.shared_state.mark_fail_open(
                f"SeaCache scheduler metadata is unavailable; running in fail-open mode: {error}"
            )
            return args, kwargs

        if (
            step_index < 0
            or num_inference_steps <= 0
            or step_index >= num_inference_steps
            or not math.isfinite(sigma)
            or not 0.0 <= sigma <= 1.0
        ):
            self.shared_state.mark_fail_open(
                "SeaCache scheduler metadata is invalid; expected a valid step index and exact sigma in [0, 1]."
            )
            return args, kwargs

        try:
            raw_vision = (
                self.raw_vision_callback(module, args, kwargs) if self.raw_vision_callback is not None else None
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self.shared_state.mark_fail_open(
                f"SeaCache model metadata is unavailable; running in fail-open mode: {error}"
            )
            return args, kwargs
        if not raw_vision:
            self.shared_state.mark_fail_open(
                "SeaCache requires raw vision latents containing at least one noisy frame; action-only, sound-only, "
                "conditioning-only, and unsupported model calls run in fail-open mode."
            )
            return args, kwargs

        self.shared_state.forward_metadata = _SeaCacheForwardMetadata(
            step_index=step_index,
            sigma=sigma,
            num_inference_steps=num_inference_steps,
            raw_vision=raw_vision,
        )
        return args, kwargs

    def prepare_decoder_stack(
        self,
        module: torch.nn.Module,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Resolve the Cosmos 3 gate eagerly before entering compiled decoder blocks."""
        unwrapped_module = unwrap_module(module)
        layers = getattr(unwrapped_module, "layers", None)
        uses_context_parallel = getattr(unwrapped_module, "_cp_shard_fn", None) is not None
        has_fsdp, has_dtensor = _parameter_sharding_types(layers) if layers else (False, False)
        uses_supported_parallelism = uses_context_parallel or has_dtensor
        synchronize_decision = (
            uses_supported_parallelism
            and not has_fsdp
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        forward_metadata = self.shared_state.forward_metadata
        if self.state_manager._current_context is None or forward_metadata is None:
            if synchronize_decision:
                _, disagreed = _synchronize_compute_decision(True, gen_seq.device)
                if disagreed and self.state_manager._current_context is not None:
                    self.state_manager.get_state().reset_trajectory()
            return und_seq, gen_seq, True

        state: SeaCacheContextState = self.state_manager.get_state()
        state.full_execution_pending = True
        state.cacheable_execution = True
        state.step_index = forward_metadata.step_index
        state.gen_input = gen_seq

        indicator_error_reported = False
        if has_fsdp:
            self.shared_state.mark_fail_open(
                "SeaCache cannot safely bypass FSDP-managed transformer blocks; running in fail-open mode."
            )
            indicator = None
            indicator_error_reported = True
        elif uses_supported_parallelism and not synchronize_decision:
            self.shared_state.mark_fail_open(
                "SeaCache requires an initialized distributed process group for Cosmos 3 context/tensor parallelism; "
                "running in fail-open mode."
            )
            indicator = None
            indicator_error_reported = True
        elif not layers:
            self.shared_state.mark_fail_open("SeaCache could not locate the Cosmos 3 decoder stack; running full.")
            indicator = None
            indicator_error_reported = True
        else:
            try:
                indicator = _build_indicator(self.config, forward_metadata)
            except (TypeError, ValueError, RuntimeError) as error:
                self.shared_state.mark_fail_open(
                    f"SeaCache could not construct its vision indicator; running in fail-open mode: {error}"
                )
                indicator = None
                indicator_error_reported = True
        if indicator is None and not indicator_error_reported:
            self.shared_state.mark_fail_open(
                "SeaCache could not construct its vision indicator; running in fail-open mode."
            )

        gate_should_compute = self.shared_state.resolve_gate(state, forward_metadata, indicator, self.config)
        should_compute = gate_should_compute
        cached_und = cached_residual = None

        if not should_compute and not state.history:
            should_compute = True
            self.shared_state.mark_fail_open(
                "SeaCache selected a cache hit without residual history; running in fail-open mode."
            )
        elif not should_compute:
            residual_history = state.history[-(self.config.residual_order + 1) :]
            _, cached_und, cached_residual = residual_history[-1]
            if (
                any(
                    residual.shape != gen_seq.shape
                    or residual.device != gen_seq.device
                    or residual.dtype != gen_seq.dtype
                    for _, _, residual in residual_history
                )
                or cached_und.shape != und_seq.shape
                or cached_und.device != und_seq.device
                or cached_und.dtype != und_seq.dtype
            ):
                state.history = []
                should_compute = True
                self.shared_state.mark_fail_open(
                    "SeaCache residual history changed shape, device, or dtype; running in fail-open mode."
                )
            elif self.config.residual_order == 1 and len(residual_history) >= 2:
                previous_step, _, previous_residual = residual_history[-2]
                latest_step, _, latest_residual = residual_history[-1]
                if latest_step != previous_step:
                    step_scale = (forward_metadata.step_index - latest_step) / (latest_step - previous_step)
                    cached_residual = latest_residual + (latest_residual - previous_residual) * step_scale

        disagreed = False
        if synchronize_decision:
            should_compute, disagreed = _synchronize_compute_decision(should_compute, gen_seq.device)

        if disagreed:
            state.reset_trajectory()
        elif should_compute and not gate_should_compute:
            state.accumulated_distance = 0.0
            state.gate_should_compute = True

        if should_compute:
            return und_seq, gen_seq, True

        state.skip_remaining = True
        state.full_execution_pending = False
        state.cached_und_output = cached_und
        state.cached_gen_residual = cached_residual
        state.consecutive_cached += 1
        return cached_und, gen_seq + cached_residual, False

    def record_decoder_stack(
        self,
        module: torch.nn.Module,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
    ) -> None:
        """Record post-normalization Cosmos 3 decoder outputs after a full step."""
        if self.state_manager._current_context is None:
            return
        state: SeaCacheContextState = self.state_manager.get_state()
        if state.full_execution_pending:
            state.und_output = und_seq
            state.gen_output = gen_seq

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        self.shared_state.forward_metadata = None
        if self.state_manager._current_context is not None:
            state: SeaCacheContextState = self.state_manager.get_state()
            if (
                self.use_stack_boundary
                and state.full_execution_pending
                and state.und_output is not None
                and state.gen_output is not None
            ):
                _record_full_execution(
                    self.config,
                    state,
                    gen_output=state.gen_output,
                    und_output=state.und_output,
                )
            else:
                state.reset_forward()
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        self.shared_state.reset()
        return module


class SeaCacheLeaderBlockHook(ModelHook):
    def __init__(
        self,
        config: SeaCacheConfig,
        state_manager: StateManager,
        shared_state: SeaCacheSharedState,
        post_norm_boundary: bool = False,
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.shared_state = shared_state
        self.post_norm_boundary = post_norm_boundary
        self._metadata = None

    def initialize_hook(self, module: torch.nn.Module):
        module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(module.__class__)
        return module

    def _build_indicator(self, forward_metadata: _SeaCacheForwardMetadata) -> list[torch.Tensor] | None:
        return _build_indicator(self.config, forward_metadata)

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        context_is_set = self.state_manager._current_context is not None
        state = self.state_manager.get_state() if context_is_set else None
        if state is not None:
            state.reset_forward()
            state.full_execution_pending = True
            state.gen_input = hidden_states

        forward_metadata = self.shared_state.forward_metadata
        if state is None or forward_metadata is None:
            return self.fn_ref.original_forward(*args, **kwargs)

        state.step_index = forward_metadata.step_index
        state.cacheable_execution = True
        indicator_error_reported = False
        if _is_parameter_sharded(module):
            self.shared_state.mark_fail_open(
                "SeaCache cannot safely bypass parameter-sharded transformer blocks; running in fail-open mode."
            )
            indicator = None
            indicator_error_reported = True
        else:
            try:
                indicator = self._build_indicator(forward_metadata)
            except (TypeError, ValueError, RuntimeError) as error:
                self.shared_state.mark_fail_open(
                    f"SeaCache could not construct its vision indicator; running in fail-open mode: {error}"
                )
                indicator = None
                indicator_error_reported = True
        if indicator is None and not indicator_error_reported:
            self.shared_state.mark_fail_open(
                "SeaCache could not construct its vision indicator; running in fail-open mode."
            )
        should_compute = self.shared_state.resolve_gate(state, forward_metadata, indicator, self.config)

        if should_compute or not state.history:
            if not should_compute:
                state.accumulated_distance = 0.0
                self.shared_state.mark_fail_open(
                    "SeaCache selected a cache hit without residual history; running in fail-open mode."
                )
            return self.fn_ref.original_forward(*args, **kwargs)

        residual_history = state.history[-(self.config.residual_order + 1) :]
        _, cached_und, cached_residual = residual_history[-1]
        expects_auxiliary_stream = self._metadata.return_encoder_hidden_states_index is not None
        if expects_auxiliary_stream:
            auxiliary_stream_mismatch = encoder_hidden_states is None or cached_und is None
        else:
            auxiliary_stream_mismatch = cached_und is not None
        if expects_auxiliary_stream and encoder_hidden_states is not None and cached_und is not None:
            auxiliary_stream_mismatch = auxiliary_stream_mismatch or (
                cached_und.shape != encoder_hidden_states.shape
                or cached_und.device != encoder_hidden_states.device
                or cached_und.dtype != encoder_hidden_states.dtype
            )
        if (
            any(
                residual.shape != hidden_states.shape
                or residual.device != hidden_states.device
                or residual.dtype != hidden_states.dtype
                for _, _, residual in residual_history
            )
            or auxiliary_stream_mismatch
        ):
            state.history = []
            state.accumulated_distance = 0.0
            self.shared_state.mark_fail_open(
                "SeaCache residual history changed shape, device, or dtype; running in fail-open mode."
            )
            return self.fn_ref.original_forward(*args, **kwargs)

        if self.config.residual_order == 1 and len(residual_history) >= 2:
            previous_step, _, previous_residual = residual_history[-2]
            latest_step, _, latest_residual = residual_history[-1]
            if latest_step != previous_step:
                step_scale = (forward_metadata.step_index - latest_step) / (latest_step - previous_step)
                cached_residual = latest_residual + (latest_residual - previous_residual) * step_scale

        state.skip_remaining = True
        state.full_execution_pending = False
        state.cached_und_output = cached_und
        state.cached_gen_residual = cached_residual
        state.consecutive_cached += 1
        if self.post_norm_boundary:
            return _build_block_output(self._metadata, hidden_states, encoder_hidden_states)
        return _build_block_output(self._metadata, hidden_states + cached_residual, cached_und)


class SeaCacheBlockHook(ModelHook):
    def __init__(
        self,
        config: SeaCacheConfig,
        state_manager: StateManager,
        shared_state: SeaCacheSharedState,
        is_tail: bool = False,
        post_norm_boundary: bool = False,
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.shared_state = shared_state
        self.is_tail = is_tail
        self.post_norm_boundary = post_norm_boundary
        self._metadata = None

    def initialize_hook(self, module: torch.nn.Module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            return self.fn_ref.original_forward(*args, **kwargs)

        state: SeaCacheContextState = self.state_manager.get_state()
        if state.skip_remaining:
            hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
            return _build_block_output(self._metadata, hidden_states, encoder_hidden_states)

        output = self.fn_ref.original_forward(*args, **kwargs)
        if not self.is_tail or state.skip_remaining or not state.full_execution_pending:
            return output
        if self.post_norm_boundary:
            return output

        hidden_states, encoder_hidden_states = _get_block_outputs(self._metadata, output)
        _record_full_execution(
            self.config,
            state,
            gen_output=hidden_states,
            und_output=encoder_hidden_states,
            requires_und_output=self._metadata.return_encoder_hidden_states_index is not None,
        )
        return output


class SeaCachePostNormHook(ModelHook):
    def __init__(
        self,
        config: SeaCacheConfig,
        state_manager: StateManager,
        shared_state: SeaCacheSharedState,
        pathway: Literal["und", "gen"],
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.shared_state = shared_state
        self.pathway = pathway

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            return self.fn_ref.original_forward(*args, **kwargs)

        state: SeaCacheContextState = self.state_manager.get_state()
        if state.skip_remaining:
            if self.pathway == "und":
                if state.cached_und_output is not None:
                    return state.cached_und_output
            elif state.gen_input is not None and state.cached_gen_residual is not None:
                output = state.gen_input + state.cached_gen_residual
                state.reset_forward()
                return output

            self.shared_state.mark_fail_open(
                "SeaCache post-normalization state is incomplete after the decoder stack was skipped."
            )
            output = self.fn_ref.original_forward(*args, **kwargs)
            state.reset_forward()
            return output

        output = self.fn_ref.original_forward(*args, **kwargs)
        if not state.full_execution_pending:
            return output
        if self.pathway == "und":
            state.und_output = output
            return output

        _record_full_execution(
            self.config,
            state,
            gen_output=output,
            und_output=state.und_output,
        )
        return output


def apply_sea_cache(module: torch.nn.Module, config: SeaCacheConfig) -> None:
    r"""
    Apply SeaCache to a supported transformer.

    The hook caches the transformer's expensive language-model hidden transform. For Cosmos 3, the cache stores
    post-normalization understanding output and a generation residual from the decoder-stack input to the
    post-normalization output. Modality prediction heads continue to run normally. Other model adapters fall back to
    caching the complete repeated-block stack.

    Args:
        module (`torch.nn.Module`):
            Transformer module to cache.
        config (`SeaCacheConfig`):
            SeaCache configuration.
    """
    from ..models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer

    unwrapped_module = unwrap_module(module)
    is_cosmos3 = isinstance(unwrapped_module, Cosmos3OmniTransformer)
    raw_vision_callback = config.raw_vision_callback
    if raw_vision_callback is None and is_cosmos3:
        raw_vision_callback = _prepare_cosmos3_raw_vision_metadata

    post_norm_modules = None
    if is_cosmos3:
        und_norm = getattr(unwrapped_module, "norm", None)
        gen_norm = getattr(unwrapped_module, "norm_moe_gen", None)
        if isinstance(und_norm, torch.nn.Module) and isinstance(gen_norm, torch.nn.Module):
            post_norm_modules = (("und", "norm", und_norm), ("gen", "norm_moe_gen", gen_norm))
        else:
            logger.warning(
                "SeaCache could not locate the Cosmos 3 final pathway normalizations; falling back to the repeated-block "
                "residual boundary."
            )
    post_norm_boundary = post_norm_modules is not None

    blocks = []
    for name, submodule in unwrapped_module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(submodule, torch.nn.ModuleList):
            continue
        blocks.extend((f"{name}.{index}", block) for index, block in enumerate(submodule))

    if not blocks:
        raise ValueError("SeaCache found no repeated transformer blocks on the model.")

    state_manager = StateManager(SeaCacheContextState)
    shared_state = SeaCacheSharedState()
    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    registrations: list[tuple[HookRegistry, str]] = []

    def register_hook(target: torch.nn.Module, hook: ModelHook, name: str) -> None:
        registry = HookRegistry.check_if_exists_or_initialize(target)
        registry.register_hook(hook, name)
        registrations.append((registry, name))

    try:
        register_hook(
            module,
            SeaCacheRootHook(
                config,
                state_manager,
                shared_state,
                raw_vision_callback,
                use_stack_boundary=is_cosmos3,
            ),
            _SEA_CACHE_ROOT_HOOK,
        )

        if not is_cosmos3:
            leader_name, leader = blocks[0]
            logger.debug(f"Applying SeaCache leader hook to '{leader_name}'.")
            register_hook(
                leader,
                SeaCacheLeaderBlockHook(config, state_manager, shared_state, post_norm_boundary=post_norm_boundary),
                _SEA_CACHE_LEADER_BLOCK_HOOK,
            )

            for name, block in blocks[1:-1]:
                logger.debug(f"Applying SeaCache identity hook to '{name}'.")
                register_hook(
                    block,
                    SeaCacheBlockHook(config, state_manager, shared_state, post_norm_boundary=post_norm_boundary),
                    _SEA_CACHE_BLOCK_HOOK,
                )

            tail_name, tail = blocks[-1]
            logger.debug(f"Applying SeaCache tail hook to '{tail_name}'.")
            register_hook(
                tail,
                SeaCacheBlockHook(
                    config,
                    state_manager,
                    shared_state,
                    is_tail=True,
                    post_norm_boundary=post_norm_boundary,
                ),
                _SEA_CACHE_BLOCK_HOOK,
            )
            if post_norm_modules is not None:
                for pathway, name, norm_module in post_norm_modules:
                    logger.debug(f"Applying SeaCache post-normalization hook to '{name}'.")
                    register_hook(
                        norm_module,
                        SeaCachePostNormHook(config, state_manager, shared_state, pathway=pathway),
                        _SEA_CACHE_POST_NORM_HOOK,
                    )
    except Exception:
        for registry, name in reversed(registrations):
            registry.remove_hook(name, recurse=False)
        root_registry._child_registries_cache = None
        raise

    root_registry._child_registries_cache = None
