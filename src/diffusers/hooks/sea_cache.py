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

import copy
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

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

    SeaCache compares Spectral-Evolution-Aware (SEA) indicators between scheduler steps. If their accumulated relative
    change stays below `threshold`, the expensive language-model hidden transform is replaced with a cached residual.
    For Cosmos 3, the residual spans the decoder stack and final pathway normalization; input packing and modality
    prediction heads still execute.

    Args:
        threshold (`float`, defaults to `0.35`):
            Accumulated relative-L1 budget. Larger values reuse the cache more often.
        residual_order (`int`, defaults to `0`):
            Order used to predict the generation-stream language-model residual. `0` directly reuses the most recent
            residual and `1` linearly extrapolates from the two most recent full executions.
        retention_steps (`int`, defaults to `1`):
            Number of initial scheduler steps that always execute in full.
        cache_end_steps (`int`, defaults to `1`):
            Number of final scheduler steps that always execute in full.
        power_exp (`float`, defaults to `3.0`):
            Exponent of the SEA clean-signal power prior. SeaCache uses `3.0` for video features.
        indicator_source (`str`, defaults to `"first_block"`):
            Feature source used to construct the SEA indicator. `"first_block"` follows the published method and
            filters the timestep-modulated pre-attention input of the first transformer block.
            `"raw_vision_latents"` is an opt-in lower-cost approximation that filters the complete raw vision latent,
            including clean conditioning frames in I2V. Thresholds are not generally transferable between the two
            sources.
        current_step_callback (`Callable[[], int]`):
            Callback returning the current scheduler step index.
        current_sigma_callback (`Callable[[], float]`):
            Callback returning the exact current scheduler sigma in `[0, 1]`.
        num_inference_steps_callback (`Callable[[], int]`):
            Callback returning the number of scheduler steps in the current pipeline call.
        metadata_callback (`Callable`, *optional*):
            Advanced model adapter returning a list of `(indices, (T, H, W))` entries that locate projected noisy
            vision tokens in the generation stream for the `"first_block"` indicator. Cosmos 3 uses its native
            adapter when this is omitted.
        raw_vision_callback (`Callable`, *optional*):
            Advanced model adapter returning raw vision latents with shape `(C, T, H, W)` for the
            `"raw_vision_latents"` indicator. Cosmos 3 uses its native adapter when this is omitted.
        gate_schedule (`Sequence[bool]`, *optional*):
            Per-step full-compute decisions to replay for controlled residual-prediction ablations. SEA still evaluates
            its natural decision and reports mismatches; invalid or unsafe calls always fail open to full compute.

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

    threshold: float = 0.35
    residual_order: int = 0
    retention_steps: int = 1
    cache_end_steps: int = 1
    power_exp: float = 3.0
    indicator_source: Literal["first_block", "raw_vision_latents"] = "first_block"
    current_step_callback: Callable[[], int] = None
    current_sigma_callback: Callable[[], float] = None
    num_inference_steps_callback: Callable[[], int] = None
    metadata_callback: Callable[
        [torch.nn.Module, tuple[Any, ...], dict[str, Any]],
        list[tuple[torch.Tensor, tuple[int, int, int]]] | None,
    ] = None
    raw_vision_callback: Callable[
        [torch.nn.Module, tuple[Any, ...], dict[str, Any]],
        list[torch.Tensor] | None,
    ] = None
    gate_schedule: Sequence[bool] = None

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError(f"`threshold` must be non-negative, got {self.threshold}.")
        if self.residual_order not in (0, 1):
            raise ValueError(f"`residual_order` must be 0 or 1, got {self.residual_order}.")
        if self.retention_steps < 0:
            raise ValueError(f"`retention_steps` must be non-negative, got {self.retention_steps}.")
        if self.cache_end_steps < 0:
            raise ValueError(f"`cache_end_steps` must be non-negative, got {self.cache_end_steps}.")
        if not math.isfinite(self.power_exp) or self.power_exp <= 0:
            raise ValueError(f"`power_exp` must be positive, got {self.power_exp}.")
        if self.indicator_source not in ("first_block", "raw_vision_latents"):
            raise ValueError(
                f"`indicator_source` must be 'first_block' or 'raw_vision_latents', got {self.indicator_source!r}."
            )
        for name in (
            "current_step_callback",
            "current_sigma_callback",
            "num_inference_steps_callback",
            "metadata_callback",
            "raw_vision_callback",
        ):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"`{name}` must be callable or `None`.")
        if self.gate_schedule is not None:
            self.gate_schedule = tuple(self.gate_schedule)
            if any(not isinstance(value, bool) for value in self.gate_schedule):
                raise TypeError("`gate_schedule` must contain only boolean values.")


@dataclass
class _SeaCacheForwardMetadata:
    step_index: int
    sigma: float
    num_inference_steps: int
    vision_layout: list[tuple[torch.Tensor, tuple[int, int, int]]] | None = None
    raw_vision: list[torch.Tensor] | None = None


class SeaCacheContextState(BaseState):
    def __init__(self):
        self.history: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.gate_key: tuple[int, float] | None = None
        self.gate_should_compute = True
        self.previous_indicator: list[torch.Tensor] | None = None
        self.accumulated_distance = 0.0
        self.skip_remaining = False
        self.full_execution_pending = False
        self.cacheable_execution = False
        self.step_index: int | None = None
        self.und_input: torch.Tensor | None = None
        self.gen_input: torch.Tensor | None = None
        self.und_output: torch.Tensor | None = None
        self.cached_und_output: torch.Tensor | None = None
        self.cached_gen_residual: torch.Tensor | None = None
        self.full_started_at: float | None = None

    def reset_forward(self):
        self.skip_remaining = False
        self.full_execution_pending = False
        self.cacheable_execution = False
        self.step_index = None
        self.und_input = None
        self.gen_input = None
        self.und_output = None
        self.cached_und_output = None
        self.cached_gen_residual = None
        self.full_started_at = None

    def reset(self):
        self.history = []
        self.gate_key = None
        self.gate_should_compute = True
        self.previous_indicator = None
        self.accumulated_distance = 0.0
        self.reset_forward()


class SeaCacheSharedState:
    def __init__(self):
        self._warned_messages: set[str] = set()
        self.reset()

    def reset(self):
        self.forward_metadata: _SeaCacheForwardMetadata | None = None
        self.transformer_calls = 0
        self.gate_evaluations = 0
        self.gate_full_decisions = 0
        self.gate_skip_decisions = 0
        self.gate_trace: list[bool] = []
        self.gate_schedule_mismatches = 0
        self.num_full_steps = 0
        self.num_cached_steps = 0
        self.fail_open_calls = 0
        self.indicator_seconds = 0.0
        self.decision_seconds = 0.0
        self.full_seconds = 0.0
        self.branch_full_executions: dict[str, int] = {}
        self.branch_reuses: dict[str, int] = {}

    def warn_once(self, message: str):
        if message not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(message)

    def mark_fail_open(self, message: str):
        self.fail_open_calls += 1
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

        self.gate_evaluations += 1
        is_non_adjacent = state.gate_key is not None and metadata.step_index != state.gate_key[0] + 1
        if is_non_adjacent:
            self.mark_fail_open("SeaCache received non-adjacent scheduler steps; running full.")
        is_retained = metadata.step_index < config.retention_steps
        is_in_cache_end = metadata.step_index >= metadata.num_inference_steps - config.cache_end_steps
        is_first_observation = state.previous_indicator is None
        invalid_gate = is_non_adjacent or indicator is None
        forced_compute = invalid_gate or is_retained or is_in_cache_end or is_first_observation
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

        if config.gate_schedule is not None:
            schedule_is_valid = len(config.gate_schedule) == metadata.num_inference_steps
            scheduled_compute = bool(config.gate_schedule[metadata.step_index]) if schedule_is_valid else True
            can_replay_skip = not (invalid_gate or is_retained or is_in_cache_end or is_first_observation)
            if not schedule_is_valid:
                self.mark_fail_open(
                    "SeaCache gate schedule length does not match the number of inference steps; running full."
                )
                should_compute = True
            elif scheduled_compute or can_replay_skip:
                if natural_should_compute != scheduled_compute:
                    self.gate_schedule_mismatches += 1
                should_compute = scheduled_compute
            else:
                self.mark_fail_open("SeaCache gate schedule requested an unsafe cache hit; running full.")
                should_compute = True

        state.accumulated_distance = 0.0 if should_compute else candidate_accumulated_distance

        state.gate_key = gate_key
        state.gate_should_compute = should_compute
        state.previous_indicator = None if indicator is None else [value.detach() for value in indicator]
        self.gate_trace.append(should_compute)
        if should_compute:
            self.gate_full_decisions += 1
        else:
            self.gate_skip_decisions += 1
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
    state_manager: StateManager,
    shared_state: SeaCacheSharedState,
    state: SeaCacheContextState,
    gen_output: torch.Tensor,
    und_output: torch.Tensor | None,
) -> None:
    shared_state.num_full_steps += 1
    branch = state_manager._current_context
    shared_state.branch_full_executions[branch] = shared_state.branch_full_executions.get(branch, 0) + 1
    if state.full_started_at is not None:
        shared_state.full_seconds += time.perf_counter() - state.full_started_at
    if (
        state.cacheable_execution
        and state.step_index is not None
        and state.gen_input is not None
        and und_output is not None
        and gen_output.shape == state.gen_input.shape
    ):
        state.history.append(
            (
                state.step_index,
                und_output.detach().clone(),
                (gen_output - state.gen_input).detach().clone(),
            )
        )
        state.history = state.history[-(config.residual_order + 1) :]
    state.reset_forward()


def _prepare_cosmos3_vision_metadata(
    module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[torch.Tensor, tuple[int, int, int]]] | None:
    module = unwrap_module(module)
    bound_arguments = inspect.signature(module.__class__.forward).bind_partial(module, *args, **kwargs).arguments
    vision_tokens = bound_arguments.get("vision_tokens")
    vision_token_shapes = bound_arguments.get("vision_token_shapes")
    vision_sequence_indexes = bound_arguments.get("vision_sequence_indexes")
    vision_timesteps = bound_arguments.get("vision_timesteps")
    vision_noisy_frame_indexes = bound_arguments.get("vision_noisy_frame_indexes")
    und_len = bound_arguments.get("und_len")

    if (
        not isinstance(vision_tokens, (list, tuple))
        or not isinstance(vision_token_shapes, (list, tuple))
        or not isinstance(vision_sequence_indexes, torch.Tensor)
        or not isinstance(vision_timesteps, torch.Tensor)
        or vision_timesteps.numel() == 0
        or not isinstance(vision_noisy_frame_indexes, (list, tuple))
        or und_len is None
        or len(vision_tokens) != len(vision_token_shapes)
        or len(vision_tokens) != len(vision_noisy_frame_indexes)
    ):
        return None

    vision_sequence_indexes = vision_sequence_indexes.flatten()
    layout = []
    offset = 0
    for token, token_shape, noisy_frame_indexes in zip(vision_tokens, vision_token_shapes, vision_noisy_frame_indexes):
        if (
            not isinstance(token, torch.Tensor)
            or not isinstance(noisy_frame_indexes, torch.Tensor)
            or len(token_shape) != 3
        ):
            return None

        temporal, height, width = (int(value) for value in token_shape)
        item_numel = temporal * height * width
        item_indexes = vision_sequence_indexes[offset : offset + item_numel]
        if item_indexes.numel() != item_numel:
            return None
        offset += item_numel

        noisy_frame_indexes = noisy_frame_indexes.flatten().to(device=item_indexes.device, dtype=torch.long)
        if noisy_frame_indexes.numel() == 0:
            continue
        if torch.any(noisy_frame_indexes < 0) or torch.any(noisy_frame_indexes >= temporal):
            return None

        item_indexes = item_indexes.reshape(temporal, height, width)
        generation_indexes = item_indexes[noisy_frame_indexes].flatten() - int(und_len)
        if torch.any(generation_indexes < 0):
            return None
        layout.append(
            (
                generation_indexes,
                (int(noisy_frame_indexes.numel()), height, width),
            )
        )

    if offset != vision_sequence_indexes.numel() or not layout:
        return None
    return layout


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


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Whether a block is managed by a parameter-sharding runtime that SeaCache cannot safely bypass."""

    for submodule in unwrap_module(module).modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            return True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            return True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if (
                parameter_type.__name__ == "FlatParameter"
                and parameter_type.__module__.startswith("torch.distributed.fsdp")
            ) or (
                parameter_type.__name__ == "DTensor"
                and parameter_type.__module__.startswith("torch.distributed.tensor")
            ):
                return True
    return False


class SeaCacheRootHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        config: SeaCacheConfig,
        state_manager: StateManager,
        shared_state: SeaCacheSharedState,
        metadata_callback: Callable,
        raw_vision_callback: Callable,
        residual_boundary: str,
    ):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.shared_state = shared_state
        self.metadata_callback = metadata_callback
        self.raw_vision_callback = raw_vision_callback
        self.residual_boundary = residual_boundary
        self._last_stats: dict[str, Any] | None = None

    @property
    def num_full_steps(self) -> int:
        return self.shared_state.num_full_steps

    @property
    def num_cached_steps(self) -> int:
        return self.shared_state.num_cached_steps

    def stats(self) -> dict[str, Any]:
        if self.shared_state.transformer_calls == 0 and self._last_stats is not None:
            return copy.deepcopy(self._last_stats)

        persistent_cache_bytes = 0
        per_branch = {}
        for branch, state in sorted(self.state_manager._state_cache.items()):
            if state.previous_indicator is not None:
                persistent_cache_bytes += sum(
                    value.numel() * value.element_size() for value in state.previous_indicator
                )
            for _, und_output, gen_residual in state.history:
                persistent_cache_bytes += und_output.numel() * und_output.element_size()
                persistent_cache_bytes += gen_residual.numel() * gen_residual.element_size()
            per_branch[branch] = {
                "full_calls": self.shared_state.branch_full_executions.get(branch, 0),
                "reuse_calls": self.shared_state.branch_reuses.get(branch, 0),
            }

        opportunities = self.shared_state.num_full_steps + self.shared_state.num_cached_steps
        return {
            "indicator_source": self.config.indicator_source,
            "residual_order": self.config.residual_order,
            "residual_boundary": self.residual_boundary,
            "transformer_calls": self.shared_state.transformer_calls,
            "gate_evaluations": self.shared_state.gate_evaluations,
            "gate_full_decisions": self.shared_state.gate_full_decisions,
            "gate_skip_decisions": self.shared_state.gate_skip_decisions,
            "gate_trace": list(self.shared_state.gate_trace),
            "gate_schedule_replayed": self.config.gate_schedule is not None,
            "gate_schedule_mismatches": self.shared_state.gate_schedule_mismatches,
            "actual_full_executions": self.shared_state.num_full_steps,
            "actual_reuses": self.shared_state.num_cached_steps,
            "actual_reuse_rate": (self.shared_state.num_cached_steps / opportunities if opportunities else 0.0),
            "fail_open_calls": self.shared_state.fail_open_calls,
            "sea_indicator_seconds": self.shared_state.indicator_seconds,
            "sea_decision_seconds": self.shared_state.decision_seconds,
            "sea_seconds": (self.shared_state.indicator_seconds + self.shared_state.decision_seconds),
            "full_seconds": self.shared_state.full_seconds,
            "timing_note": "host wall time; CUDA work is asynchronous",
            "persistent_cache_bytes": persistent_cache_bytes,
            "branch_full_executions": dict(sorted(self.shared_state.branch_full_executions.items())),
            "branch_reuses": dict(sorted(self.shared_state.branch_reuses.items())),
            "per_branch": per_branch,
        }

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        self.shared_state.forward_metadata = None
        self.shared_state.transformer_calls += 1
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

        vision_layout = None
        raw_vision = None
        try:
            if self.config.indicator_source == "first_block":
                vision_layout = (
                    self.metadata_callback(module, args, kwargs) if self.metadata_callback is not None else None
                )
            else:
                raw_vision = (
                    self.raw_vision_callback(module, args, kwargs) if self.raw_vision_callback is not None else None
                )
        except (TypeError, ValueError, RuntimeError) as error:
            self.shared_state.mark_fail_open(
                f"SeaCache model metadata is unavailable; running in fail-open mode: {error}"
            )
            return args, kwargs
        if self.config.indicator_source == "first_block" and not vision_layout:
            self.shared_state.mark_fail_open(
                "SeaCache requires noisy vision tokens; action-only, sound-only, and conditioning-only calls run in "
                "fail-open mode."
            )
            return args, kwargs
        if self.config.indicator_source == "raw_vision_latents" and not raw_vision:
            self.shared_state.mark_fail_open(
                "SeaCache requires raw noisy vision latents for the selected indicator source; action-only, sound-only, "
                "conditioning-only, and unsupported model calls run in fail-open mode."
            )
            return args, kwargs

        self.shared_state.forward_metadata = _SeaCacheForwardMetadata(
            step_index=step_index,
            sigma=sigma,
            num_inference_steps=num_inference_steps,
            vision_layout=vision_layout,
            raw_vision=raw_vision,
        )
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        self.shared_state.forward_metadata = None
        if self.state_manager._current_context is not None:
            self.state_manager.get_state().reset_forward()
        return output

    def reset_state(self, module: torch.nn.Module):
        if self.shared_state.transformer_calls > 0:
            self._last_stats = self.stats()
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
        self._normalization = None

    def initialize_hook(self, module: torch.nn.Module):
        module = unwrap_module(module)
        self._metadata = TransformerBlockRegistry.get(module.__class__)
        if self._metadata.hidden_states_norm_module_name is not None:
            self._normalization = getattr(module, self._metadata.hidden_states_norm_module_name)
        return module

    def _build_indicator(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: _SeaCacheForwardMetadata,
    ) -> list[torch.Tensor] | None:
        if self.config.indicator_source == "raw_vision_latents":
            if not forward_metadata.raw_vision:
                return None
            indicator = []
            for latent in forward_metadata.raw_vision:
                raw_vision = latent.movedim(0, -1)
                indicator.append(
                    _apply_sea_filter(
                        raw_vision,
                        sigma=forward_metadata.sigma,
                        power_exp=self.config.power_exp,
                    ).detach()
                )
            return indicator

        if self._normalization is None:
            return None
        if not forward_metadata.vision_layout:
            return None
        normalized_hidden_states = self._normalization(hidden_states)
        indicator = []
        for indexes, shape in forward_metadata.vision_layout:
            indexes = indexes.to(device=normalized_hidden_states.device, dtype=torch.long)
            if (
                indexes.numel() != math.prod(shape)
                or torch.any(indexes < 0)
                or torch.any(indexes >= normalized_hidden_states.shape[0])
            ):
                return None
            noisy_vision = normalized_hidden_states.index_select(0, indexes).reshape(*shape, -1)
            indicator.append(
                _apply_sea_filter(
                    noisy_vision,
                    sigma=forward_metadata.sigma,
                    power_exp=self.config.power_exp,
                ).detach()
            )
        return indicator

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        context_is_set = self.state_manager._current_context is not None
        state = self.state_manager.get_state() if context_is_set else None
        if state is not None:
            state.reset_forward()
            state.full_execution_pending = True
            state.gen_input = hidden_states
            state.und_input = encoder_hidden_states

        forward_metadata = self.shared_state.forward_metadata
        if state is None or forward_metadata is None:
            if state is not None:
                state.full_started_at = time.perf_counter()
            return self.fn_ref.original_forward(*args, **kwargs)

        state.step_index = forward_metadata.step_index
        state.cacheable_execution = True
        indicator_started = time.perf_counter()
        indicator_error_reported = False
        if _is_parameter_sharded(module):
            self.shared_state.mark_fail_open(
                "SeaCache cannot safely bypass parameter-sharded transformer blocks; running in fail-open mode."
            )
            indicator = None
            indicator_error_reported = True
        else:
            try:
                indicator = self._build_indicator(hidden_states, forward_metadata)
            except (TypeError, ValueError, RuntimeError) as error:
                self.shared_state.mark_fail_open(
                    f"SeaCache could not construct its vision indicator; running in fail-open mode: {error}"
                )
                indicator = None
                indicator_error_reported = True
        self.shared_state.indicator_seconds += time.perf_counter() - indicator_started
        if indicator is None and not indicator_error_reported:
            self.shared_state.mark_fail_open(
                "SeaCache could not construct its vision indicator; running in fail-open mode."
            )
        decision_started = time.perf_counter()
        should_compute = self.shared_state.resolve_gate(state, forward_metadata, indicator, self.config)
        self.shared_state.decision_seconds += time.perf_counter() - decision_started

        if should_compute or not state.history:
            if not should_compute:
                state.accumulated_distance = 0.0
                self.shared_state.mark_fail_open(
                    "SeaCache selected a cache hit without residual history; running in fail-open mode."
                )
            state.full_started_at = time.perf_counter()
            return self.fn_ref.original_forward(*args, **kwargs)

        residual_history = state.history[-(self.config.residual_order + 1) :]
        _, cached_und, cached_residual = residual_history[-1]
        if (
            any(
                residual.shape != hidden_states.shape
                or residual.device != hidden_states.device
                or residual.dtype != hidden_states.dtype
                for _, _, residual in residual_history
            )
            or encoder_hidden_states is None
            or cached_und.shape != encoder_hidden_states.shape
            or cached_und.device != encoder_hidden_states.device
            or cached_und.dtype != encoder_hidden_states.dtype
        ):
            state.history = []
            state.accumulated_distance = 0.0
            self.shared_state.mark_fail_open(
                "SeaCache residual history changed shape, device, or dtype; running in fail-open mode."
            )
            state.full_started_at = time.perf_counter()
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
        self.shared_state.num_cached_steps += 1
        branch = self.state_manager._current_context
        self.shared_state.branch_reuses[branch] = self.shared_state.branch_reuses.get(branch, 0) + 1
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
            self.state_manager,
            self.shared_state,
            state,
            gen_output=hidden_states,
            und_output=encoder_hidden_states,
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
            self.state_manager,
            self.shared_state,
            state,
            gen_output=output,
            und_output=state.und_output,
        )
        return output


def apply_sea_cache(module: torch.nn.Module, config: SeaCacheConfig) -> None:
    r"""
    Apply SeaCache to a supported transformer.

    The hook caches the transformer's expensive language-model hidden transform. For Cosmos 3, the cache stores
    post-normalization understanding output and a generation residual from the pre-block input to the
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
    metadata_callback = config.metadata_callback
    raw_vision_callback = config.raw_vision_callback
    if metadata_callback is None or raw_vision_callback is None:
        if is_cosmos3:
            if metadata_callback is None:
                metadata_callback = _prepare_cosmos3_vision_metadata
            if raw_vision_callback is None:
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
    residual_boundary = "post_language_model_norm" if post_norm_boundary else "repeated_block_stack"

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
                metadata_callback,
                raw_vision_callback,
                residual_boundary,
            ),
            _SEA_CACHE_ROOT_HOOK,
        )

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


def get_sea_cache_stats(module: torch.nn.Module) -> dict[str, Any]:
    """Return statistics for the SeaCache instance currently attached to ``module``."""

    registry = getattr(module, "_diffusers_hook", None)
    root_hook = registry.get_hook(_SEA_CACHE_ROOT_HOOK) if registry is not None else None
    if not isinstance(root_hook, SeaCacheRootHook):
        raise ValueError("SeaCache is not enabled on this module.")
    return root_hook.stats()
