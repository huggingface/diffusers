# Copyright 2025 Alpha-VLLM and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LuminaFlowMatchSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class LuminaFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Rectified Flow scheduler for Lumina-T2I.
    
    This scheduler implements the rectified flow matching used in Lumina, which learns a velocity field
    that transports samples from a noise distribution to a data distribution along straight paths.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`]
    and [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The time shift factor for sampling. Higher values shift the distribution towards the end.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to use dynamic time shifting based on image resolution.
        base_image_seq_len (`int`, defaults to 256):
            Base sequence length for dynamic shifting calculation.
        max_image_seq_len (`int`, defaults to 4096):
            Maximum sequence length for dynamic shifting calculation.
        base_shift (`float`, defaults to 0.5):
            Base shift value for dynamic shifting.
        max_shift (`float`, defaults to 1.15):
            Maximum shift value for dynamic shifting.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        # Initialize timesteps
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _apply_time_shift(self, timesteps: torch.Tensor, image_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply time shifting to timesteps.
        
        Args:
            timesteps: The timesteps to shift.
            image_seq_len: Image sequence length for dynamic shifting.
        
        Returns:
            Shifted timesteps.
        """
        if self.config.use_dynamic_shifting and image_seq_len is not None:
            # Calculate shift based on image resolution
            shift = self.config.base_shift + (self.config.max_shift - self.config.base_shift) * (
                image_seq_len - self.config.base_image_seq_len
            ) / (self.config.max_image_seq_len - self.config.base_image_seq_len)
            shift = max(self.config.base_shift, min(shift, self.config.max_shift))
        else:
            shift = self.config.shift

        # Apply shift: t_shifted = t / (t + shift * (1 - t))
        if shift != 1.0:
            timesteps = timesteps / (timesteps + shift * (1.0 - timesteps))
        
        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[torch.Tensor] = None,
        image_seq_len: Optional[int] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`torch.Tensor`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is
                passed, `num_inference_steps` must be `None`.
            image_seq_len (`int`, *optional*):
                Image sequence length for dynamic time shifting.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must provide either `num_inference_steps` or `timesteps`.")
        
        if timesteps is not None:
            self.timesteps = timesteps.to(device)
            self.num_inference_steps = len(timesteps)
        else:
            self.num_inference_steps = num_inference_steps
            
            # Create linear timesteps from 0 to 1
            timesteps = torch.linspace(0.0, 1.0, num_inference_steps, dtype=torch.float32)
            
            # Apply time shifting
            timesteps = self._apply_time_shift(timesteps, image_seq_len)
            
            self.timesteps = timesteps.to(device=device)

        self._step_index = None
        self._begin_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[LuminaFlowMatchSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model (velocity prediction).
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_utils.LuminaFlowMatchSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.LuminaFlowMatchSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.LuminaFlowMatchSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Get current and next timesteps
        t = timestep
        if isinstance(t, torch.Tensor):
            t = t.to(sample.device)
        
        # Calculate step size (dt)
        if self._step_index < len(self.timesteps) - 1:
            dt = self.timesteps[self._step_index + 1] - t
        else:
            dt = 1.0 - t

        if isinstance(dt, torch.Tensor):
            dt = dt.to(sample.device)
        elif not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, device=sample.device, dtype=sample.dtype)

        # Euler step: x_{t+dt} = x_t + v_t * dt
        # where v_t is the velocity predicted by the model
        prev_sample = sample + model_output * dt

        # Update step index
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return LuminaFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def _init_step_index(self, timestep):
        """
        Initialize the step index counter.
        """
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = (self.timesteps == timestep).nonzero().item()
        else:
            self._step_index = self._begin_index

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the rectified flow formulation.
        
        For rectified flow: x_t = (1 - t) * noise + t * x_0
        
        Args:
            original_samples (`torch.Tensor`):
                The original samples (x_0).
            noise (`torch.Tensor`):
                The noise to add (x_1, usually Gaussian).
            timesteps (`torch.Tensor`):
                The timesteps for each sample.
        
        Returns:
            `torch.Tensor`: The noisy samples.
        """
        # Ensure timesteps are on the same device as samples
        timesteps = timesteps.to(original_samples.device)
        
        # Reshape timesteps to match sample dimensions
        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)
        
        # Linear interpolation: x_t = (1 - t) * noise + t * x_0
        noisy_samples = (1.0 - timesteps) * noise + timesteps * original_samples
        
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the velocity target for training.
        
        For rectified flow, the velocity is: v = x_0 - x_1 = x_0 - noise
        
        Args:
            sample (`torch.Tensor`):
                The original sample (x_0).
            noise (`torch.Tensor`):
                The noise sample (x_1).
            timesteps (`torch.Tensor`):
                The timesteps (not used in rectified flow, but kept for interface compatibility).
        
        Returns:
            `torch.Tensor`: The velocity target.
        """
        return sample - noise

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        """
        Get the previous timestep.
        """
        if self.step_index is not None and self.step_index < len(self.timesteps) - 1:
            return self.timesteps[self.step_index + 1]
        return timestep

