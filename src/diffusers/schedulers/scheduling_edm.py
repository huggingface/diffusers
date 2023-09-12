# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging, randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->KarrasEDM
class KarrasEDMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class KarrasEDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements the "EDM" column of Table 1 and Algorithm 2 in Karras et al. 2022 [1].

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`, *optional*, defaults t0 40):
            Number of diffusion steps used to train the model.
        prediction_type (`str`, *optional*, defaults to `edm`):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        timestep_spacing (`str`, *optional*, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        precondition_type (`str`, *optional*, defaults to `edm`):
            The preconditioning type for the model, which determines the preconditioning scalings used. See the
            "Network and precondtioning" section of Table 1 in [1]. Currently only `edm` is supported.
        sigma_min (`float`, *optional*, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 100]. (TODO: is this correct?)
        sigma_data (`float`, *optional*, defaults to 0.5):
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        s_churn (`float`, *optional*, defaults to 0.0):
            The parameter controlling the overall amount of stochasticity if we add noise during sampling. Defaults to
            0.0; a reasonable range is [0, 100].
        s_tmin (`float`, *optional*, defaults to 0.0):
            The start value of the sigma range where we add noise. Defaults to 0.0; a reasonable range is [0, 10].
        s_tmax (`float`, *optional*, defaults to `float('int')`):
            The end value of the sigma range where we add noise. Defaults to `float('inf')`; a reasonable range is
            [0.2, float('inf')].
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample. This counteracts the loss of detail during sampling. Defaults
            to 0.0; a reasonable range is [1.000, 1.011].
        rho (`float`, *optional*, defaults to 7.0):
            The rho parameter used for calculating the Karras sigma schedule, which is set to 7.0 in the EDM paper [1].
        clip_sample (`bool`, *optional*, defaults to `True`):
            Whether to clip the predicted sample for numerical stability.
        clip_sample_range (`float`, *optional*, defaults to `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
    """

    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        prediction_type: str = "edm",
        timestep_spacing: str = "linspace",
        precondition_type: str = "edm",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        rho: float = 7.0,
        clip_sample: Optional[bool] = False,
        clip_sample_range: float = 1.0,
    ):
        # setable values
        self.precondition_type = precondition_type
        self.set_timesteps(num_train_timesteps, None, None, num_train_timesteps)

        self.custom_timesteps = False
        self.is_scale_input_called = False

        self._step_index = None

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.config.sigma_max

        return self.config.sigma_max

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index

    @property
    def state_in_first_order(self):
        return self.dt is None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]

        self._step_index = step_index.item()

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(self._index_counter) == 0:
            pos = 1 if len(indices) > 1 else 0
        else:
            timestep_int = timestep.cpu().item() if torch.is_tensor(timestep) else timestep
            pos = self._index_counter[timestep_int]

        return indices[pos].item()

    def precondition_inputs(self, sample, sigma):
        if self.precondition_type in ["edm", "cm_edm"]:
            scaled_sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        else:
            # No input preconditioning
            scaled_sample = sample

        return scaled_sample

    def precondition_noise(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])

        if self.precondition_type == "edm":
            scaled_noise = 0.25 * torch.log(sigma)
        elif self.precondition_type == "cm_edm":
            scaled_noise = 1000 * 0.25 * torch.log(sigma + 1e-44)
        else:
            # No noise preconditioning.
            scaled_noise = sigma

        return scaled_noise

    def precondition_outputs(self, sample, model_output, sigma):
        if self.precondition_type in ["edm", "cm_edm"]:
            sigma_data = self.config.sigma_data
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5

            denoised = c_skip * sample + c_out * model_output
        else:
            # No output preconditioning
            denoised = model_output

        return denoised

    def scale_model_input(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, Union[float, torch.FloatTensor]]:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Preconditions the sample input to the neural network $F_\theta$, following c_in(sigma) for the EDM column in
        Table 1.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`float` or `torch.FloatTensor`):
                The current timestep in the diffusion chain.
        Returns:
            `torch.FloatTensor`:
                A scaled input sample and timestep.
        """
        # 1. Get sigma and sigma_hat corresponding to timestep
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_hat = self.sigma_hats[self.step_index]

        # 2. Add noise based on noise level sigma_hat
        if self.state_in_first_order:
            if sigma_hat > sigma:  # equivalent to gamma > 0
                eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
                sample = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

            self.sample_hat = sample

        # 3. Precondition the input sample and timestep.
        sample = self.precondition_inputs(sample, sigma_hat)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, optional):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, optional):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.
            num_train_timesteps: (`int`, *optional*):
                Number of diffusion steps used to train the model.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        num_train_timesteps = num_train_timesteps or self.config.num_train_timesteps

        # 1. Handle custom timestesp or generate timesteps
        # NOTE: Follows DDPMScheduler custom timesteps logic
        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("Custom `timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
            elif self.config.timestep_spacing == "leading":
                step_ratio = num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        # 2. Calculate the Karras sigmas
        if self.custom_timesteps:
            # timesteps is in decreasing order, but ramp should be in increasing order
            ramp = timesteps[::-1].copy() / (self.config.num_train_timesteps - 1)
        else:
            ramp = np.linspace(0, 1, self.num_inference_steps)

        sigmas = self._convert_to_karras(ramp)

        # timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        # 3. Calculate the sigma_hat schedule for stochastic sampling
        sigma_hats = self._get_sigma_hats(sigmas)

        # 4. Calculate timestep schedule from sigmas and sigma_hats
        # In the sampling loop, we want to output timesteps in the following order:
        # [sigma_hat_0, sigma_1, sigma_hat_1, sigma_2, ..., sigma_hat_{n - 1}]
        timesteps = np.empty((sigma_hats.size + sigmas.size - 1,), dtype=np.float32)
        timesteps[0::2] = sigma_hats
        timesteps[1::2] = sigmas[1:]

        # 5. Finish processing sigmas and timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(device=device)
        # Magic to convert sigmas from [sigma_1, sigma_2, ..., sigma_n, 0.0] to
        # [sigma_1, sigma_2, sigma_2, sigma_3, sigma_3, ..., sigma_n, sigma_n, 0.0]
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        sigma_hats = np.concatenate([sigma_hats, [0.0]]).astype(np.float32)
        sigma_hats = torch.from_numpy(sigma_hats).to(device=device)
        # Same schedule as self.sigmas:
        # [sigma_hat_1, sigma_hat_2, sigma_hat_2, sigma_hat_3, sigma_hat_3, ..., sigma_hat_n, sigma_hat_n, 0.0]
        self.sigma_hats = torch.cat([sigma_hats[:1], sigma_hats[1:-1].repeat_interleave(2), sigma_hats[-1:]])

        timesteps = torch.from_numpy(timesteps)
        timesteps = self.precondition_noise(timesteps)
        self.timesteps = timesteps.to(device=device)

        # 6. Empty dt and derivative to set the scheduler in first order mode
        self.prev_derivative = None
        self.dt = None
        self.sample_hat = None

        # 7. Reset step_index
        self._step_index = None

        # (YiYi Notes: keep this for now since we are keeping add_noise function which use index_for_timestep)
        # for exp beta schedules, such as the one for `pipeline_shap_e.py`
        # we need an index counter
        self._index_counter = defaultdict(int)

    # Modified _convert_to_karras implementation that takes in ramp as argument
    def _convert_to_karras(self, ramp, sigma_min=None, sigma_max=None):
        """Constructs the noise schedule of Karras et al. (2022)."""
        if sigma_min is None:
            sigma_min = self.config.sigma_min
        if sigma_max is None:
            sigma_max = self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _get_sigma_hats(self, sigmas: np.ndarray) -> np.ndarray:
        # TODO: vectorize this?
        sigma_hats = []
        for sigma in sigmas:
            # Temporarily raise the noise level (see Line 5 of Algorithm 2 in the EDM paper)
            if self.config.s_tmin <= sigma <= self.config.s_tmax:
                gamma = min(self.config.s_churn / len(sigmas), 2**0.5 - 1)
            else:
                gamma = 0
            sigma_hat = sigma * (gamma + 1)
            sigma_hats.append(sigma_hat)
        sigma_hats = np.asarray(sigma_hats, dtype=sigmas.dtype)
        return sigma_hats

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[KarrasEDMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This takes an Euler step to reverse the ODE from sigma_hat to sigma (the Karras sigma corresponding to
        timestep).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_utils.KarrasEDMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.KarrasEDMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        # 0. Check inputs
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    f"Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    f" `{self.__class__}.step()` is not supported. Make sure to pass"
                    f" one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        # 1. Initialize step_index if necessary
        if self.step_index is None:
            self._init_step_index(timestep)

        # 2. Get current sigmas
        # (YiYi notes: keep this for now since we are keeping the add_noise method)
        # advance index counter by 1
        timestep_int = timestep.cpu().item() if torch.is_tensor(timestep) else timestep
        self._index_counter[timestep_int] += 1

        if self.state_in_first_order:
            # 1st order / Euler's method
            # sigma = self.sigmas[self.step_index]
            sigma_hat = self.sigma_hats[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
        else:
            # 2nd order / Heun's method
            # sigma = self.sigmas[self.step_index - 1]
            sigma_hat = self.sigma_hats[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

        # hack: Set sample_hat to sample in case it's None
        if self.sample_hat is None:
            self.sample_hat = sample

        # 3. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        sigma_input = sigma_hat if self.state_in_first_order else sigma_next
        sample_input = self.sample_hat if self.state_in_first_order else sample
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample_input - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            # sample * c_out + input * c_skip
            # TODO: how should this interact with self.precondition_outputs below?
            pred_original_sample = model_output * (-sigma_input / (sigma_input**2 + 1) ** 0.5) + (
                sample_input / (sigma_input**2 + 1)
            )
        elif self.config.prediction_type == "edm":
            # Apply output preconditioning
            pred_original_sample = self.precondition_outputs(sample_input, model_output, sigma_input)
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Perform a first order (Euler) or second order correction (Heun) step.
        if self.state_in_first_order:
            # 4.1. 1st order / Euler's method
            # 4.l.1. Get Karras ODE derivative (Line 7 in Algorithm 2 in EDM paper)
            derivative = (self.sample_hat - pred_original_sample) / sigma_hat  # d_i

            # 4.1.2. Get delta timestep
            dt = sigma_next - sigma_hat

            # 4.1.3. Take Euler step (Line 8 in Algorithm 2 in EDM paper)
            prev_sample = self.sample_hat + derivative * dt  # x_{i + 1}

            # 4.1.4. Store values for 2nd order step
            self.prev_derivative = derivative
            self.dt = dt
        else:
            # 4.2. 2nd order / Heun's method
            # 4.2.1. Get Karras ODE derivative (Line 10 in Algorithm 2 in EDM paper)
            # NOTE: sample here corresponds to x_{i + 1} in Algorithm 2, which is the output of the Euler step from
            # the previous scheduler step()
            derivative = (sample - pred_original_sample) / sigma_next  # d_i'
            # 4.2.2 Get Heun correction to the derivative
            derivative = (self.prev_derivative + derivative) / 2

            # 4.2.3. Take Heun step (Line 11 in Algorithm 2 in EDM paper)
            prev_sample = self.sample_hat + derivative * self.dt

            # 4.2.4. Put the scheduler in first order mode by freeing up dt and derivative
            self.prev_derivative = None
            self.dt = None
            self.sample_hat = None

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return KarrasEDMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # TODO: change to match noise added for EDM training???
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
