# # Copyright 2024 Sana-Sprint Authors and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..schedulers.scheduling_utils import SchedulerMixin
from ..utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->SCM
class SCMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class SCMScheduler(SchedulerMixin, ConfigMixin):
    """
    `SCMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance. This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass
    documentation for the generic methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        prediction_type (`str`, defaults to `trigflow`):
            Prediction type of the scheduler function. Currently only supports "trigflow".
        max_timesteps (`float`, defaults to 1.57080):
            The maximum timestep value used in the diffusion process.
        intermediate_timesteps (`float`, *optional*, defaults to 1.3):
            The intermediate timestep value used when num_inference_steps=2.
        sigma_data (`float`, defaults to 0.5):
            The standard deviation of the noise added during multi-step inference.
    """

    # _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "trigflow",
        max_timesteps: float = 1.57080,
        intermediate_timesteps: Optional[float] = 1.3,
        sigma_data: float = 0.5,
    ):
        """
        Initialize the SCM scheduler.

        Args:
            num_train_timesteps (`int`, defaults to 1000):
                The number of diffusion steps to train the model.
            prediction_type (`str`, defaults to `trigflow`):
                Prediction type of the scheduler function. Currently only supports "trigflow".
            max_timesteps (`float`, defaults to 1.57080):
                The maximum timestep value used in the diffusion process.
            intermediate_timesteps (`float`, *optional*, defaults to 1.3):
                The intermediate timestep value used when num_inference_steps=2.
            sigma_data (`float`, defaults to 0.5):
                The standard deviation of the noise added during multi-step inference.
        """
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def set_timesteps(
        self,
        num_inference_steps: int,
        timesteps: torch.Tensor = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        if timesteps is not None and len(timesteps) == num_inference_steps + 1:
            if isinstance(timesteps, list):
                self.timesteps = torch.tensor(timesteps, device=device).float()
            elif isinstance(timesteps, torch.Tensor):
                self.timesteps = timesteps.to(device).float()
            else:
                raise ValueError(f"Unsupported timesteps type: {type(timesteps)}")
        elif self.config.intermediate_timesteps and num_inference_steps == 2:
            self.timesteps = torch.tensor(
                [self.config.max_timesteps, self.config.intermediate_timesteps, 0], device=device
            ).float()
        elif self.config.intermediate_timesteps:
            self.timesteps = torch.linspace(
                self.config.max_timesteps, 0, num_inference_steps + 1, device=device
            ).float()
            logger.warning(
                f"Intermediate timesteps for SCM is not supported when num_inference_steps != 2. "
                f"Reset timesteps to {self.timesteps} default max_timesteps"
            )
        else:
            # max_timesteps=arctan(80/0.5)=1.56454 is the default from sCM paper, we choose a different value here
            self.timesteps = torch.linspace(
                self.config.max_timesteps, 0, num_inference_steps + 1, device=device
            ).float()

        print(f"Set timesteps: {self.timesteps}")

    def step(
        self,
        model_output: torch.FloatTensor,
        timeindex: int,
        timestep: float,
        sample: torch.FloatTensor,
        generator: torch.Generator = None,
        return_dict: bool = True,
    ) -> Union[SCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.SCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_scm.SCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 2. compute alphas, betas
        t = self.timesteps[timeindex + 1]
        s = self.timesteps[timeindex]

        # 4. Different Parameterization:
        parameterization = self.config.prediction_type

        if parameterization == "trigflow":
            pred_x0 = torch.cos(s) * sample - torch.sin(s) * model_output
        else:
            raise ValueError(f"Unsupported parameterization: {parameterization}")

        # 5. Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
            noise = (
                torch.randn(model_output.shape, device=model_output.device, generator=generator)
                * self.config.sigma_data
            )
            prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise
        else:
            prev_sample = pred_x0

        if not return_dict:
            return (prev_sample, pred_x0)

        return SCMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)

    def __len__(self):
        return self.config.num_train_timesteps
