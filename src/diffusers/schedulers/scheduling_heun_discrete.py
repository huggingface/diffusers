# Copyright 2023 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
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

import math
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .. utils import logging, randn_tensor
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class HeunDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler with Heun steps for discrete beta schedules.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper) or `edm` (apply output preconditioning).
        interpolation_type(`str`, defaults to `"linear"`, *optional*):
            The interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be on of
            `"linear"` or `"log_linear"`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product like in Stable
            Diffusion.
        precondition_type (`str`, *optional*, defaults to `None`):
            The preconditioning type for the model, which determines the preconditioning scalings used. See the
            "Network and precondtioning" section of Table 1 in [1]. If `None`, no preconditioning is performed.
        sigma_min (`float`, *optional*, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 100]. (TODO: is this correct?)
        sigma_data (`float`, *optional*, defaults to 1.0):
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
        stochastic_sampling (`bool`, *optional*, defaults to `False`):
            Whether to perform stochastic sampling according to Algorithm 2 in the EDM paper.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,  # sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        clip_sample: Optional[bool] = False,
        clip_sample_range: float = 1.0,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        precondition_type: Optional[str] = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 1.0,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        rho: float = 7.0,
        stochastic_sampling: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="cosine")
        elif beta_schedule == "exp":
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="exp")
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        #  set all values
        self.precondition_type = precondition_type
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps, None)
        self.use_karras_sigmas = use_karras_sigmas

        self.custom_timesteps = False
        self.is_scale_input_called = False

        self._step_index = None

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

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        return (self.sigmas.max() ** 2 + self.config.sigma_data ** 2) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index
    
    def precondition_inputs(self, sample, sigma):
        # Always precondition inputs...?
        scaled_sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)

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
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_hat = self.sigma_hats[self.step_index]

        if self.state_in_first_order:
            if self.config.stochastic_sampling and sigma_hat > sigma:  # equivalent to gamma > 0
                eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
                sample = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

                self.sample_hat = sample
            
            # self.sample_hat = sample
        
        sample = self.precondition_inputs(sample, sigma)
        # NOTE: the current implementation gets sample_hat after input preconditioning, but it seems like Algorithm
        # 2 wants to get sample_hat before preconditioning
        if self.sample_hat is None:
            self.sample_hat = sample

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        num_train_timesteps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            num_train_timesteps (`int`, defaults to 1000):
                The number of diffusion steps to train the model. If not set, this will default to
                `self.config.num_train_timesteps`.
            timesteps (`List[int]`, optional):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.
        """
        # 0. Check inputs and set values
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

        # 2. Define sigmas and handle sigma interpolation
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        elif self.config.interpolation_type == "log_linear":
            sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp()
        else:
            raise ValueError(
                f"{self.config.interpolation_type} is not implemented. Please specify interpolation_type to either"
                " 'linear' or 'log_linear'"
            )
        
        # 3. Calculate the Karras sigmas, if necessary
        if self.config.use_karras_sigmas:
            if self.custom_timesteps:
                # timesteps is in decreasing order, but ramp should be in increasing order
                ramp = timesteps[::-1].copy() / (self.config.num_train_timesteps - 1)
                sigma_min = self.config.sigma_min
                sigma_max = self.config.sigma_max
            else:
                ramp = np.linspace(0, 1, self.num_inference_steps)
                sigma_min: float = sigmas[-1].item()
                sigma_max: float = sigmas[0].item()
            
            # sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            sigmas = self._convert_to_karras(ramp, sigma_min, sigma_max)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        
        # 4. Calculate sigma_hat schedule
        # TODO: vectorize this?
        sigma_hats = []
        for sigma in sigmas:
            # Temporarily raise the noise level (see Line 5 of Algorithm 2 in the EDM paper)
            if self.config.stochastic_sampling and (self.config.s_tmin <= sigma <= self.config.s_tmax):
                gamma = min(self.config.s_churn / len(sigmas), 2**0.5 - 1)
            else:
                gamma = 0
            sigma_hat = sigma * (gamma + 1)
            sigma_hats.append(sigma_hat)
        sigma_hats = np.asarray(sigma_hats, dtype=sigmas.dtype)

        # 5. Calculate timestep schedule from sigmas and sigma_hats
        # TODO: fix the condition here (should also be applicable when we're not using custom timesteps?)
        if self.custom_timesteps:
            # In the sampling loop, we want to output timesteps in the following order:
            # [sigma_hat_0, sigma_1, sigma_hat_1, sigma_2, ..., sigma_hat_{n - 1}, 0]
            timesteps = np.empty((sigma_hats.size + sigmas.size - 1,), dtype=sigmas.dtype)
            timesteps[0::2] = sigma_hats
            timesteps[1::2] = sigmas[1:]

        # 6. Finish processing sigmas and timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(device=device)
        # [sigma_0, sigma_1, sigma_2, ..., sigma_{n - 1}, 0] ->
        # [sigma_0, sigma_1, sigma_1, sigma_2, sigma_2, ...,sigma_{n - 1}, sigma_{n - 1}, 0]
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        sigma_hats = np.concatenate([sigma_hats, [0.0]]).astype(np.float32)
        sigma_hats = torch.from_numpy(sigma_hats).to(device=device)
        # Make sigma_hats have the same schedule as self.sigmas
        self.sigma_hats = torch.cat([sigma_hats[:1], sigma_hats[1:-1].repeat_interleave(2), sigma_hats[-1:]])

        timesteps = torch.from_numpy(timesteps)
        timesteps = self.precondition_noise(timesteps)
        if not self.custom_timesteps:
            # TODO: for now keep this in, need to figure out logic when timesteps are set with a beta schedule
            # [t_0, t_1, t_2, ..., t_{n - 1}] -> [t_0, t_1, t_1, t_2, t_2, ..., t_{n - 1}, t_{n - 1}]
            timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])

        self.timesteps = timesteps.to(device=device)

        # 7. Empty dt and derivative to set scheduler to first order mode
        self.prev_derivative = None
        self.dt = None
        self.sample_hat = None

        # 8. Reset _step_index
        self._step_index = None

        # (YiYi Notes: keep this for now since we are keeping add_noise function which use index_for_timestep)
        # for exp beta schedules, such as the one for `pipeline_shap_e.py`
        # we need an index counter
        self._index_counter = defaultdict(int)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
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

    # # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    # def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
    #     """Constructs the noise schedule of Karras et al. (2022)."""

    #     sigma_min: float = in_sigmas[-1].item()
    #     sigma_max: float = in_sigmas[0].item()

    #     rho = 7.0  # 7.0 is the value used in the paper
    #     ramp = np.linspace(0, 1, num_inference_steps)
    #     min_inv_rho = sigma_min ** (1 / rho)
    #     max_inv_rho = sigma_max ** (1 / rho)
    #     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    #     return sigmas

    # Modified _convert_to_karras implementation that takes in ramp as argument
    def _convert_to_karras(self, ramp, sigma_min, sigma_max):
        """Constructs the noise schedule of Karras et al. (2022)."""
        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

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

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: Union[float, torch.FloatTensor],
        sample: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        # 0. Check inputs
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
            sigma = self.sigmas[self.step_index]
            sigma_hat = self.sigma_hats[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
        else:
            # 2nd order / Heun's method
            sigma = self.sigmas[self.step_index - 1]
            sigma_hat = self.sigma_hats[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

        # currently only gamma=0 is supported. This usually works best anyways.
        # We can support gamma in the future but then need to scale the timestep before
        # passing it to the model which requires a change in API
        # gamma = 0
        # sigma_hat = sigma * (gamma + 1)  # Note: sigma_hat == sigma for now

        # TODO: Hack for tests that call step() without first calling scale_model_input()???
        if self.sample_hat is None:
            self.sample_hat = sample

        # 3. compute predicted original sample (x_0) from sigma-scaled predicted noise
        sigma_input = sigma_hat if self.state_in_first_order else sigma_next
        sample_input = self.sample_hat if self.state_in_first_order else sample
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample_input - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma_input / (sigma_input**2 + 1) ** 0.5) + (
                sample_input / (sigma_input**2 + 1)
            )
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "edm":
            # Get denoiser output based on EDM output preconditioning type.
            pred_original_sample = self.precondition_outputs(sample_input, pred_original_sample, sigma_input)
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
            derivative = (sample - pred_original_sample) / sigma_hat
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
            derivative = (sample - pred_original_sample) / sigma_next
            # 4.2.2 Get Heun correction to the derivative
            derivative = (self.prev_derivative + derivative) / 2

            # 4.2.3. Take Heun step (Line 11 in Algorithm 2 in EDM paper)
            prev_sample = self.sample_hat + derivative * self.dt

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.prev_derivative = None
            self.dt = None
            self.sample_hat = None

        # prev_sample = sample + derivative * dt

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

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
