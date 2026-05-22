# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMapEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor`):
            Computed sample :math:`z_r` at the target flow-map timestep `r_timestep`. Should be used as the next
            denoising input.
    """

    prev_sample: torch.Tensor


class FlowMapEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler-style sampler for flow-map-distilled diffusion models.

    Flow-map models learn arbitrary-interval transitions :math:`z_t \\to z_r` rather than the fixed :math:`z_t \\to
    z_0` mapping of consistency models, so a single distilled checkpoint can be evaluated at 1, 2, 4, 8, ... NFE
    without retraining. The `step` method advances the sample from `timestep` to `r_timestep` along the predicted
    velocity.

    Introduced in [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map
    Distillation](https://huggingface.co/papers/2605.13724) by Yuchao Gu, Guian Fang et al.

    This scheduler inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods implemented for all schedulers (loading, saving, etc.).

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps used to train the underlying flow-map model.
        shift (`float`, defaults to 1.0):
            Multiplicative timestep shift applied to the inference schedule. ``shift=1.0`` is the identity; values
            greater than 1.0 push the schedule toward more denoising at later steps (e.g., ``shift=5`` matches the
            Wan2.1 default).
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        # `_step_index` and `_begin_index` mirror `FlowMatchEulerDiscreteScheduler`'s state machine:
        # `_step_index` advances on every `step()` so callbacks and composable schedulers can read it;
        # `_begin_index` is honoured on the very first `step()` after `set_timesteps` to support
        # mid-schedule restarts (e.g. image-to-image style use).
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None
        self.set_timesteps(num_train_timesteps, device="cpu")

    @property
    def step_index(self) -> Optional[int]:
        """The index counter for current timestep. Returns ``None`` before the first :meth:`step` call after
        :meth:`set_timesteps`."""
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """The index for the first timestep — set by :meth:`set_begin_index`. Defaults to ``None``."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Set the begin index for the scheduler. Pipelines that start mid-schedule (e.g. image-to-image)
        call this between :meth:`set_timesteps` and the first :meth:`step` to anchor the rollout."""
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """No-op identity scaling. Provided for API compatibility with other Diffusers schedulers."""
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Linearly interpolate ``sample`` toward ``noise`` according to the normalized ``timestep``."""
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        sample = timestep * noise + (1.0 - timestep) * sample
        return sample

    def apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply the configured shift transformation to a sigma tensor."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        timesteps: Optional[List[float]] = None,
    ) -> None:
        """Build the inference timestep schedule.

        Internally tracks ``self.sigmas`` of length ``num_inference_steps + 1`` (the configured shift applied to a
        linspace from ``1.0`` to ``0.0`` by default); ``self.timesteps`` exposes the first ``num_inference_steps``
        sigmas scaled by ``num_train_timesteps`` — i.e. one timestep per inference step, matching
        :class:`~diffusers.schedulers.FlowMatchEulerDiscreteScheduler`. The final sigma (``0``) is the implicit
        r-endpoint of the last step and is appended automatically when ``sigmas`` / ``timesteps`` are user-provided.

        Args:
            num_inference_steps (`int`, *optional*):
                Number of inference steps. If ``None``, must pass ``sigmas`` or ``timesteps``.
            device (`str` or `torch.device`, *optional*):
                Target device for ``self.sigmas`` / ``self.timesteps``.
            sigmas (`List[float]`, *optional*):
                Custom sigma schedule of length ``num_inference_steps``. The terminal ``0`` sigma is appended
                automatically. The configured ``shift`` is applied on top.
            timesteps (`List[float]`, *optional*):
                Custom timestep schedule of length ``num_inference_steps``, in the same units as ``self.timesteps``
                (i.e. scaled by ``num_train_timesteps``). Converted to sigmas internally. If both ``sigmas`` and
                ``timesteps`` are passed, their lengths must match.
        """
        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise ValueError("`sigmas` and `timesteps` should have the same length")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as `num_inference_steps` when both are provided"
                )
        elif sigmas is not None:
            num_inference_steps = len(sigmas)
        elif timesteps is not None:
            num_inference_steps = len(timesteps)
        else:
            raise ValueError("`num_inference_steps` must be provided when both `sigmas` and `timesteps` are `None`")

        # MPS / NPU don't support float64 — build the schedule in float64 on CPU and only move
        # the final tensors to the requested device (with a float32 downcast for MPS / NPU).
        device_obj = torch.device(device) if device is not None and not isinstance(device, torch.device) else device
        is_mps = device_obj is not None and device_obj.type == "mps"
        is_npu = device_obj is not None and device_obj.type == "npu"
        out_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        # Build the working sigma sequence (length N) before appending the terminal 0.
        if sigmas is not None:
            working_sigmas = torch.tensor(sigmas, dtype=torch.float64)
        elif timesteps is not None:
            working_sigmas = torch.tensor(timesteps, dtype=torch.float64) / self.config.num_train_timesteps
        else:
            working_sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64)[:-1]

        working_sigmas = self.apply_shift(working_sigmas)
        # Append the terminal 0 sigma as the r-endpoint of the last step.
        full_sigmas = torch.cat([working_sigmas, torch.zeros(1, dtype=working_sigmas.dtype)])

        self.num_inference_steps = num_inference_steps
        self.sigmas = full_sigmas.to(device=device, dtype=out_dtype)
        self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(device=device, dtype=out_dtype)
        # Reset the state machine — first `step()` after this will re-initialize `_step_index`.
        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep: Union[float, torch.FloatTensor]) -> None:
        """Initialize ``self._step_index`` on the first :meth:`step` call after :meth:`set_timesteps`.

        Off-schedule timesteps are allowed (any-step sampling is documented in :meth:`step`); in that case the counter
        starts at 0 so it can still be used as an observable rollout marker.
        """
        if self._begin_index is not None:
            self._step_index = self._begin_index
            return
        idx = self.index_for_timestep(timestep)
        self._step_index = idx if idx is not None else 0

    def index_for_timestep(self, timestep: Union[float, torch.FloatTensor]) -> Optional[int]:
        """Return the index of ``timestep`` on the current schedule, or ``None`` if off-schedule.

        Lookup is done against ``self.timesteps`` with a small fp tolerance. Used to recover the corresponding sigma
        without assuming the linear ``timesteps = sigmas * num_train_timesteps`` relationship — that way a custom
        schedule (e.g. non-linear shift, manually-set timesteps) still resolves correctly.
        """
        if self.timesteps is None:
            return None
        t_value = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        diffs = (self.timesteps.float() - t_value).abs()
        idx = int(diffs.argmin().item())
        if diffs[idx].item() > 1e-3:
            return None
        return idx

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        r_timestep: Optional[Union[float, torch.FloatTensor]] = None,
        return_dict: bool = True,
    ) -> Union[FlowMapEulerDiscreteSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Advance ``sample`` from ``timestep`` to ``r_timestep`` using the model-predicted velocity.

        Unlike a standard Euler scheduler, both endpoints of the interval can be caller-provided so that any-step
        sampling is possible: a single model call can step from `t` to any chosen target `r` (including `r=0` for a
        one-shot generation). When ``r_timestep`` is omitted, it defaults to the next timestep on the schedule
        (matching ``FlowMatchEulerDiscreteScheduler`` semantics).

        Internally the source and target sigmas are recovered by indexing ``self.sigmas`` via
        :meth:`index_for_timestep` rather than by dividing the input timesteps by ``num_train_timesteps``, so any
        schedule whose timestep / sigma relationship is non-linear (for example a custom shift) stays correct. For an
        off-schedule ``r_timestep``, the scheduler falls back to ``r_timestep / num_train_timesteps`` so any-step
        sampling outside the schedule remains supported.

        Args:
            model_output (`torch.Tensor`):
                Direct output from the flow-map model (predicted mean velocity).
            timestep (`float` or `torch.Tensor`):
                Source timestep ``t`` in the same units as ``self.timesteps``.
            sample (`torch.Tensor`):
                Current sample :math:`z_t`.
            r_timestep (`float` or `torch.Tensor`, *optional*):
                Target timestep ``r``. Defaults to the next timestep on the schedule when ``None``; pass an explicit
                value for any-step sampling. ``r_timestep == timestep`` is a no-op.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`FlowMapEulerDiscreteSchedulerOutput`] (the default) or a plain tuple.

        Returns:
            [`FlowMapEulerDiscreteSchedulerOutput`] or `tuple`:
                When ``return_dict=True``, returns a [`FlowMapEulerDiscreteSchedulerOutput`] whose ``prev_sample`` is
                :math:`z_r`. Otherwise returns a 1-tuple ``(prev_sample,)``.
        """
        if self.sigmas is None or self.timesteps is None:
            raise ValueError("`set_timesteps` has not been called.")

        # `_step_index` is maintained purely as observable state for callbacks / composable schedulers.
        # Sigma resolution stays a pure function of the passed-in (`timestep`, `r_timestep`) so the call is
        # idempotent — calling `step` twice with the same arguments always returns the same `prev_sample`.
        if self._step_index is None:
            self._init_step_index(timestep)

        # Resolve source sigma via index lookup; fall back to / num_train_timesteps only if `timestep` is off-schedule.
        t_idx = self.index_for_timestep(timestep)
        if t_idx is not None:
            sigma_t = self.sigmas[t_idx].to(device=sample.device, dtype=self.sigmas.dtype)
        else:
            t_value = timestep.to(self.sigmas.dtype) if torch.is_tensor(timestep) else torch.tensor(timestep)
            sigma_t = (t_value / self.config.num_train_timesteps).to(device=sample.device, dtype=self.sigmas.dtype)

        # Resolve target sigma. None defaults to sigmas[t_idx + 1] when on-schedule; otherwise the caller's
        # explicit `r_timestep` is used (sigma lookup first, fall back to scaling for off-schedule any-step).
        if r_timestep is None:
            if t_idx is None:
                raise ValueError(
                    "`r_timestep` is None but `timestep` is not on the current schedule, so `r` cannot be inferred. "
                    "Please pass an explicit `r_timestep` for any-step sampling outside the schedule."
                )
            sigma_r = self.sigmas[t_idx + 1].to(device=sample.device, dtype=self.sigmas.dtype)
        else:
            r_idx = self.index_for_timestep(r_timestep)
            if r_idx is not None:
                sigma_r = self.sigmas[r_idx].to(device=sample.device, dtype=self.sigmas.dtype)
            else:
                r_value = r_timestep.to(self.sigmas.dtype) if torch.is_tensor(r_timestep) else torch.tensor(r_timestep)
                sigma_r = (r_value / self.config.num_train_timesteps).to(device=sample.device, dtype=self.sigmas.dtype)

        sigma_t = sigma_t.view(*sigma_t.shape, *([1] * (model_output.ndim - sigma_t.ndim)))
        sigma_r = sigma_r.view(*sigma_r.shape, *([1] * (model_output.ndim - sigma_r.ndim)))
        prev_sample = sample - (sigma_t - sigma_r) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        # Advance state machine so downstream callbacks / composable schedulers observe correct `step_index`.
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMapEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
