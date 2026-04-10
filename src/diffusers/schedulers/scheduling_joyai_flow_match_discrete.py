# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from typing import Union

import torch

from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


class JoyAIFlowMatchDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
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
        self.register_to_config(reverse=reverse)

    def sd3_time_shift(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (self.config.shift * timesteps) / (1 + (self.config.shift - 1) * timesteps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
        **kwargs,
    ):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas.to(device=device)
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self._step_index = None
