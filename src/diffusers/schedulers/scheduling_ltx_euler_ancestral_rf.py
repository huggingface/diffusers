# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

"""
LTXEulerAncestralRFScheduler

This scheduler implements a K-diffusion style Euler-Ancestral sampler specialized for flow / CONST parameterization,
closely mirroring ComfyUI's `sample_euler_ancestral_RF` implementation used for LTX-Video.

Reference implementation (ComfyUI):
    comfy.k_diffusion.sampling.sample_euler_ancestral_RF
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LTXEulerAncestralRFSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor`):
            Updated sample for the next step in the denoising process.
    """

    prev_sample: torch.FloatTensor


class LTXEulerAncestralRFScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler-Ancestral scheduler for LTX-Video (RF / CONST parametrization).

    This scheduler is intended for models where the network is trained with a CONST-like parameterization (as in LTXV /
    FLUX). It approximates ComfyUI's `sample_euler_ancestral_RF` sampler and is useful when reproducing ComfyUI
    workflows inside diffusers.

    The scheduler can either:
    - reuse the [`FlowMatchEulerDiscreteScheduler`] sigma / timestep logic when only `num_inference_steps` is provided
      (default diffusers-style usage), or
    - follow an explicit ComfyUI-style sigma schedule when `sigmas` (or `timesteps`) are passed to [`set_timesteps`].

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Included for config compatibility; not used to build the schedule.
        eta (`float`, defaults to 1.0):
            Stochasticity parameter. `eta=0.0` yields deterministic DDIM-like sampling; `eta=1.0` matches ComfyUI's
            default RF behavior.
        s_noise (`float`, defaults to 1.0):
            Global scaling factor for the stochastic noise term.
    """

    # Allow config migration from the flow-match scheduler and back.
    _compatibles = ["FlowMatchEulerDiscreteScheduler"]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        eta: float = 1.0,
        s_noise: float = 1.0,
    ):
        # Note: num_train_timesteps is kept only for config compatibility.
        self.num_inference_steps: Optional[int] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """
        The index for the first timestep. It can be set from a pipeline with `set_begin_index` to support
        image-to-image like workflows that start denoising part-way through the schedule.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Included for API compatibility; not strictly needed here but kept to allow pipelines that call
        `set_begin_index`.
        """
        self._begin_index = begin_index

    def index_for_timestep(
        self, timestep: Union[float, torch.Tensor], schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:
        """
        Map a (continuous) `timestep` value to an index into `self.timesteps`.

        This follows the convention used in other discrete schedulers: if the same timestep value appears multiple
        times in the schedule (which can happen when starting in the middle of the schedule), the *second* occurrence
        is used for the first `step` call so that no sigma is accidentally skipped.
        """
        if schedule_timesteps is None:
            if self.timesteps is None:
                raise ValueError("Timesteps have not been set. Call `set_timesteps` first.")
            schedule_timesteps = self.timesteps

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(schedule_timesteps.device)

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        if len(indices) == 0:
            raise ValueError(
                "Passed `timestep` is not in `self.timesteps`. Make sure to use values from `scheduler.timesteps`."
            )

        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.Tensor]):
        """
        Initialize the internal step index based on a given timestep.
        """
        if self.timesteps is None:
            raise ValueError("Timesteps have not been set. Call `set_timesteps` first.")

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        sigmas: Optional[Union[List[float], torch.Tensor]] = None,
        timesteps: Optional[Union[List[float], torch.Tensor]] = None,
        mu: Optional[float] = None,
        **kwargs,
    ):
        """
        Set the sigma / timestep schedule for sampling.

        When `sigmas` or `timesteps` are provided explicitly, they are used as the RF sigma schedule (ComfyUI-style)
        and are expected to include the terminal 0.0. When both are `None`, the scheduler reuses the
        [`FlowMatchEulerDiscreteScheduler`] logic to generate sigmas from `num_inference_steps` and the stored config
        (including any resolution-dependent shifting, Karras/beta schedules, etc.).

        Args:
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. If provided together with explicit `sigmas`/`timesteps`, they are expected
                to be consistent and are otherwise ignored with a warning.
            device (`str` or `torch.device`, *optional*):
                Device to move the internal tensors to.
            sigmas (`List[float]` or `torch.Tensor`, *optional*):
                Explicit sigma schedule, e.g. `[1.0, 0.99, ..., 0.0]`.
            timesteps (`List[float]` or `torch.Tensor`, *optional*):
                Optional alias for `sigmas`. If `sigmas` is None and `timesteps` is provided, timesteps are treated as
                sigmas.
            mu (`float`, *optional*):
                Optional shift parameter used when delegating to [`FlowMatchEulerDiscreteScheduler.set_timesteps`] and
                `config.use_dynamic_shifting` is `True`.
        """
        # 1. Auto-generate schedule (FlowMatch-style) when no explicit sigmas/timesteps are given
        if sigmas is None and timesteps is None:
            if num_inference_steps is None:
                raise ValueError(
                    "LTXEulerAncestralRFScheduler.set_timesteps requires either explicit `sigmas`/`timesteps` "
                    "or a `num_inference_steps` value."
                )

            # We reuse FlowMatchEulerDiscreteScheduler to construct a sigma schedule that is
            # consistent with the original LTX training setup (including optional time shifting,
            # Karras / exponential / beta schedules, etc.).
            from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

            base_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.config)
            base_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=None,
                mu=mu,
                timesteps=None,
            )

            self.num_inference_steps = base_scheduler.num_inference_steps
            # Keep sigmas / timesteps on the requested device so step() can operate on-device without
            # extra transfers.
            self.sigmas = base_scheduler.sigmas.to(device=device)
            self.timesteps = base_scheduler.timesteps.to(device=device)
            self._step_index = None
            self._begin_index = None
            return

        # 2. Explicit sigma schedule (ComfyUI-style path)
        if sigmas is None:
            # `timesteps` is treated as sigmas in RF / flow-matching setups.
            sigmas = timesteps

        if isinstance(sigmas, list):
            sigmas_tensor = torch.tensor(sigmas, dtype=torch.float32)
        elif isinstance(sigmas, torch.Tensor):
            sigmas_tensor = sigmas.to(dtype=torch.float32)
        else:
            raise TypeError(f"`sigmas` must be a list or torch.Tensor, got {type(sigmas)}.")

        if sigmas_tensor.ndim != 1:
            raise ValueError(f"`sigmas` must be a 1D tensor, got shape {tuple(sigmas_tensor.shape)}.")

        if sigmas_tensor[-1].abs().item() > 1e-6:
            logger.warning(
                "The last sigma in the schedule is not zero (%.6f). "
                "For best compatibility with ComfyUI's RF sampler, the terminal sigma "
                "should be 0.0.",
                sigmas_tensor[-1].item(),
            )

        # Move to device once, then derive timesteps.
        if device is not None:
            sigmas_tensor = sigmas_tensor.to(device)

        # Internal sigma schedule stays in [0, 1] (as provided).
        self.sigmas = sigmas_tensor
        # Timesteps are scaled to match the training setup of LTX (FlowMatch-style),
        # where the network expects timesteps on [0, num_train_timesteps].
        # This keeps the transformer conditioning in the expected range while the RF
        # scheduler still operates on the raw sigma values.
        num_train = float(getattr(self.config, "num_train_timesteps", 1000))
        self.timesteps = sigmas_tensor * num_train

        if num_inference_steps is not None and num_inference_steps != len(sigmas) - 1:
            logger.warning(
                "Provided `num_inference_steps=%d` does not match `len(sigmas)-1=%d`. "
                "Overriding `num_inference_steps` with `len(sigmas)-1`.",
                num_inference_steps,
                len(sigmas) - 1,
            )

        self.num_inference_steps = len(sigmas) - 1
        self._step_index = None
        self._begin_index = None

    def _sigma_broadcast(self, sigma: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Helper to broadcast a scalar sigma to the shape of `sample`.
        """
        while sigma.ndim < sample.ndim:
            sigma = sigma.view(*sigma.shape, 1)
        return sigma

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LTXEulerAncestralRFSchedulerOutput, Tuple[torch.FloatTensor]]:
        """
        Perform a single Euler-Ancestral RF update step.

        Args:
            model_output (`torch.FloatTensor`):
                Raw model output at the current step. Interpreted under the CONST parametrization as `v_t`, with
                denoised state reconstructed as `x0 = x_t - sigma_t * v_t`.
            timestep (`float` or `torch.Tensor`):
                The current sigma value (must match one entry in `self.timesteps`).
            sample (`torch.FloatTensor`):
                Current latent sample `x_t`.
            generator (`torch.Generator`, *optional*):
                Optional generator for reproducible noise.
            return_dict (`bool`):
                If `True`, return a `LTXEulerAncestralRFSchedulerOutput`; otherwise return a tuple where the first
                element is the updated sample.
        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `LTXEulerAncestralRFScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` values as `timestep`."
                ),
            )

        if self.sigmas is None or self.timesteps is None:
            raise ValueError("Scheduler has not been initialized. Call `set_timesteps` before `step`.")

        if self._step_index is None:
            self._init_step_index(timestep)

        i = self._step_index
        if i >= len(self.sigmas) - 1:
            # Already at the end; simply return the current sample.
            prev_sample = sample
        else:
            # Work in float32 for numerical stability
            sample_f = sample.to(torch.float32)
            model_output_f = model_output.to(torch.float32)

            sigma = self.sigmas[i]
            sigma_next = self.sigmas[i + 1]

            sigma_b = self._sigma_broadcast(sigma.view(1), sample_f)
            sigma_next_b = self._sigma_broadcast(sigma_next.view(1), sample_f)

            # Approximate denoised x0 under CONST parametrization:
            #   x0 = x_t - sigma_t * v_t
            denoised = sample_f - sigma_b * model_output_f

            if sigma_next.abs().item() < 1e-8:
                # Final denoising step
                x = denoised
            else:
                eta = float(self.config.eta)
                s_noise = float(self.config.s_noise)

                # Downstep computation (ComfyUI RF variant)
                downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
                sigma_down = sigma_next * downstep_ratio

                alpha_ip1 = 1.0 - sigma_next
                alpha_down = 1.0 - sigma_down

                # Deterministic part (Euler step in (x, x0)-space)
                sigma_down_b = self._sigma_broadcast(sigma_down.view(1), sample_f)
                alpha_ip1_b = self._sigma_broadcast(alpha_ip1.view(1), sample_f)
                alpha_down_b = self._sigma_broadcast(alpha_down.view(1), sample_f)

                sigma_ratio = sigma_down_b / sigma_b
                x = sigma_ratio * sample_f + (1.0 - sigma_ratio) * denoised

                # Stochastic ancestral noise
                if eta > 0.0 and s_noise > 0.0:
                    renoise_coeff = (
                        (sigma_next_b**2 - sigma_down_b**2 * alpha_ip1_b**2 / (alpha_down_b**2 + 1e-12))
                        .clamp(min=0.0)
                        .sqrt()
                    )

                    noise = randn_tensor(
                        sample_f.shape, generator=generator, device=sample_f.device, dtype=sample_f.dtype
                    )
                    x = (alpha_ip1_b / (alpha_down_b + 1e-12)) * x + noise * renoise_coeff * s_noise

            prev_sample = x.to(sample.dtype)

        # Advance internal step index
        self._step_index = min(self._step_index + 1, len(self.sigmas) - 1)

        if not return_dict:
            return (prev_sample,)

        return LTXEulerAncestralRFSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        # For compatibility with other schedulers; used e.g. in some training
        # utilities to infer the maximum number of training timesteps.
        return int(getattr(self.config, "num_train_timesteps", 1000))
