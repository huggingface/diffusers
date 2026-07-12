# Copyright 2025 HuggingFace Inc.
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
Ping-pong scheduler for Stable Audio 3 (distilled rectified flow).

Reference: ``sample_flow_pingpong`` in
  stable_audio_3/inference/sampling.py
"""

from typing import Literal, Optional, Union

import torch

from ..configuration_utils import register_to_config
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


class PingPongScheduler(FlowMatchEulerDiscreteScheduler):
    """
    Ping-pong sampler for Stable Audio 3's distilled rectified-flow model.

    Each step applies two sub-operations:

    1. **Denoise** — predict x̂₀ from the model's velocity output: ``x̂₀ = x_t − t · v(x_t, t)``
    2. **Re-noise** — draw fresh noise and interpolate toward the next sigma: ``x_{t+1} = (1 − σ_{i+1}) · x̂₀ + σ_{i+1}
       · ε``, ``ε ~ N(0, I)``

    The "ping-pong" character is the fresh noise draw — unlike DDIM, the noise direction is resampled each step rather
    than being inferred from the model. This is exactly [`FlowMatchEulerDiscreteScheduler`]'s
    ``stochastic_sampling=True`` update rule, so this class only supplies the SA3-specific noise schedule (via
    `set_timesteps`) and reuses the parent's ``step``.

    The schedule is logSNR-uniform: ``N+1`` breakpoints equally spaced in log-SNR space ``[logsnr_min, logsnr_max]``,
    converted to the flow-matching *t* variable via ``t = sigmoid(−λ)``. Because ``sigmoid`` is decreasing, the first
    sigma is the highest (most noisy) and the last is the lowest.

    Args:
        num_inference_steps (`int`, defaults to 8):
            Number of denoising steps. SA3 Medium is distilled for exactly 8.
        logsnr_min (`float`, defaults to −6.2):
            Minimum log-SNR value — maps to the high-noise start of the schedule.
        logsnr_max (`float`, defaults to 2.0):
            Maximum log-SNR value — maps to the low-noise end of the schedule.

    The remaining arguments configure the inherited [`FlowMatchEulerDiscreteScheduler`] machinery and should be left at
    their defaults for SA3 — ``num_train_timesteps=1`` and ``shift=1.0`` make the parent's *t* variable equal to the
    sigma directly (no rescaling to ``[0, 1000]`` or shift-warping), matching SA3's timestep embedding, and
    ``stochastic_sampling=True`` selects the ping-pong (denoise + re-noise) update rule.
    """

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 8,
        logsnr_min: float = -6.2,
        logsnr_max: float = 2.0,
        num_train_timesteps: int = 1,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
        stochastic_sampling: bool = True,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=invert_sigmas,
            shift_terminal=shift_terminal,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
            stochastic_sampling=stochastic_sampling,
        )

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigma_max: float = 1.0,
    ) -> None:
        """
        Build the logSNR-uniform noise schedule and store it.

        Args:
            num_inference_steps (`int`, *optional*):
                Override the configured step count.
            device (`str` or `torch.device`, *optional*):
                Device for the schedule tensors.
            sigma_max (`float`, defaults to 1.0):
                Compresses the schedule to ``[sigma_max, 0]`` instead of ``[1, 0]``, while keeping the full step count.
                Used for audio-to-audio variation (``init_noise_level``): a lower `sigma_max` starts the denoising loop
                closer to the clean signal, so less of the original audio is altered.
        """
        n = num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps

        logsnr = torch.linspace(self.config.logsnr_min, self.config.logsnr_max, n + 1)
        sigmas = torch.sigmoid(-logsnr)  # (N+1,), decreasing
        sigmas[0] = 1.0  # sigma_max: start from pure noise
        sigmas[-1] = 0.0  # end fully denoised
        sigmas = sigmas * sigma_max

        super().set_timesteps(num_inference_steps=n, sigmas=sigmas[:-1].tolist(), device=device)

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Identity — SA3 does not pre-condition model inputs."""
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: ``x_t = (1 − t)·x₀ + t·ε``.

        Consistent with the rectified-flow interpolant used by Stable Audio 3.
        """
        t = timesteps.to(dtype=original_samples.dtype, device=original_samples.device)
        while t.ndim < original_samples.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * original_samples + t * noise

    def __len__(self) -> int:
        return self.config.num_inference_steps
