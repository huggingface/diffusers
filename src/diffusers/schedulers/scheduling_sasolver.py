# Copyright 2024 Shuchen Xue, etc. in University of Chinese Academy of Sciences Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: check https://arxiv.org/abs/2309.05019
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py

import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import deprecate, is_scipy_available
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


if is_scipy_available():
    import scipy.stats


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
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class SASolverScheduler(SchedulerMixin, ConfigMixin):
    """
    `SASolverScheduler` is a fast dedicated high-order solver for diffusion SDEs.

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
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        predictor_order (`int`, defaults to 2):
            The predictor order which can be `1` or `2` or `3` or '4'. It is recommended to use `predictor_order=2` for
            guided sampling, and `predictor_order=3` for unconditional sampling.
        corrector_order (`int`, defaults to 2):
            The corrector order which can be `1` or `2` or `3` or '4'. It is recommended to use `corrector_order=2` for
            guided sampling, and `corrector_order=3` for unconditional sampling.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        tau_func (`Callable`, *optional*):
            Stochasticity during the sampling. Default in init is `lambda t: 1 if t >= 200 and t <= 800 else 0`.
            SA-Solver will sample from vanilla diffusion ODE if tau_func is set to `lambda t: 0`. SA-Solver will sample
            from vanilla diffusion SDE if tau_func is set to `lambda t: 1`. For more details, please check
            https://arxiv.org/abs/2309.05019
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++"`.
        algorithm_type (`str`, defaults to `data_prediction`):
            Algorithm type for the solver; can be `data_prediction` or `noise_prediction`. It is recommended to use
            `data_prediction` with `solver_order=2` for guided sampling like in Stable Diffusion.
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Default = True.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        use_beta_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use beta sigmas for step sizes in the noise schedule during the sampling process. Refer to [Beta
            Sampling is All You Need](https://huggingface.co/papers/2407.12173) for more information.
        lambda_min_clipped (`float`, defaults to `-inf`):
            Clipping threshold for the minimum value of `lambda(t)` for numerical stability. This is critical for the
            cosine (`squaredcos_cap_v2`) noise schedule.
        variance_type (`str`, *optional*):
            Set to "learned" or "learned_range" for diffusion models that predict variance. If set, the model's output
            contains the predicted Gaussian variance.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        predictor_order: int = 2,
        corrector_order: int = 2,
        prediction_type: str = "epsilon",
        tau_func: Optional[Callable] = None,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "data_prediction",
        lower_order_final: bool = True,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        use_flow_sigmas: Optional[bool] = False,
        flow_shift: Optional[float] = 1.0,
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        if algorithm_type not in ["data_prediction", "noise_prediction"]:
            raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.timestep_list = [None] * max(predictor_order, corrector_order - 1)
        self.model_outputs = [None] * max(predictor_order, corrector_order - 1)

        if tau_func is None:
            self.tau_func = lambda t: 1 if t >= 200 and t <= 800 else 0
        else:
            self.tau_func = tau_func
        self.predict_x0 = algorithm_type == "data_prediction"
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

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

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
        last_timestep = ((self.config.num_train_timesteps - clipped_idx).numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            )

        elif self.config.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        if self.config.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        elif self.config.use_exponential_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        elif self.config.use_beta_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        elif self.config.use_flow_sigmas:
            alphas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
            sigmas = 1.0 - alphas
            sigmas = np.flip(self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas))[:-1].copy()
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)
        self.model_outputs = [
            None,
        ] * max(self.config.predictor_order, self.config.corrector_order - 1)
        self.lower_order_nums = 0
        self.last_sample = None

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

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

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the data_prediction/noise_prediction algorithm needs.
        Noise_prediction is designed to discretize an integral of the noise prediction model, and data_prediction is
        designed to discretize an integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either data_prediction or noise_prediction for both
        noise prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # SA-Solver_data_prediction needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["data_prediction"]:
            if self.config.prediction_type == "epsilon":
                # SA-Solver only needs the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    "`v_prediction`, or `flow_prediction` for the SASolverScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # SA-Solver_noise_prediction needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["noise_prediction"]:
            if self.config.prediction_type == "epsilon":
                # SA-Solver only needs the "mean" output.
                if self.config.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.config.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.config.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the SASolverScheduler."
                )

            if self.config.thresholding:
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def get_coefficients_exponential_negative(self, order, interval_start, interval_end):
        """
        Calculate the integral of exp(-x) * x^order dx from interval_start to interval_end
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        if order == 0:
            return torch.exp(-interval_end) * (torch.exp(interval_end - interval_start) - 1)
        elif order == 1:
            return torch.exp(-interval_end) * (
                (interval_start + 1) * torch.exp(interval_end - interval_start) - (interval_end + 1)
            )
        elif order == 2:
            return torch.exp(-interval_end) * (
                (interval_start**2 + 2 * interval_start + 2) * torch.exp(interval_end - interval_start)
                - (interval_end**2 + 2 * interval_end + 2)
            )
        elif order == 3:
            return torch.exp(-interval_end) * (
                (interval_start**3 + 3 * interval_start**2 + 6 * interval_start + 6)
                * torch.exp(interval_end - interval_start)
                - (interval_end**3 + 3 * interval_end**2 + 6 * interval_end + 6)
            )

    def get_coefficients_exponential_positive(self, order, interval_start, interval_end, tau):
        """
        Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # after change of variable(cov)
        interval_end_cov = (1 + tau**2) * interval_end
        interval_start_cov = (1 + tau**2) * interval_start

        if order == 0:
            return (
                torch.exp(interval_end_cov) * (1 - torch.exp(-(interval_end_cov - interval_start_cov))) / (1 + tau**2)
            )
        elif order == 1:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov - 1)
                    - (interval_start_cov - 1) * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 2)
            )
        elif order == 2:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**2 - 2 * interval_end_cov + 2)
                    - (interval_start_cov**2 - 2 * interval_start_cov + 2)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 3)
            )
        elif order == 3:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**3 - 3 * interval_end_cov**2 + 6 * interval_end_cov - 6)
                    - (interval_start_cov**3 - 3 * interval_start_cov**2 + 6 * interval_start_cov - 6)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 4)
            )

    def lagrange_polynomial_coefficient(self, order, lambda_list):
        """
        Calculate the coefficient of lagrange polynomial
        """

        assert order in [0, 1, 2, 3]
        assert order == len(lambda_list) - 1
        if order == 0:
            return [[1]]
        elif order == 1:
            return [
                [
                    1 / (lambda_list[0] - lambda_list[1]),
                    -lambda_list[1] / (lambda_list[0] - lambda_list[1]),
                ],
                [
                    1 / (lambda_list[1] - lambda_list[0]),
                    -lambda_list[0] / (lambda_list[1] - lambda_list[0]),
                ],
            ]
        elif order == 2:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1])
            return [
                [
                    1 / denominator1,
                    (-lambda_list[1] - lambda_list[2]) / denominator1,
                    lambda_list[1] * lambda_list[2] / denominator1,
                ],
                [
                    1 / denominator2,
                    (-lambda_list[0] - lambda_list[2]) / denominator2,
                    lambda_list[0] * lambda_list[2] / denominator2,
                ],
                [
                    1 / denominator3,
                    (-lambda_list[0] - lambda_list[1]) / denominator3,
                    lambda_list[0] * lambda_list[1] / denominator3,
                ],
            ]
        elif order == 3:
            denominator1 = (
                (lambda_list[0] - lambda_list[1])
                * (lambda_list[0] - lambda_list[2])
                * (lambda_list[0] - lambda_list[3])
            )
            denominator2 = (
                (lambda_list[1] - lambda_list[0])
                * (lambda_list[1] - lambda_list[2])
                * (lambda_list[1] - lambda_list[3])
            )
            denominator3 = (
                (lambda_list[2] - lambda_list[0])
                * (lambda_list[2] - lambda_list[1])
                * (lambda_list[2] - lambda_list[3])
            )
            denominator4 = (
                (lambda_list[3] - lambda_list[0])
                * (lambda_list[3] - lambda_list[1])
                * (lambda_list[3] - lambda_list[2])
            )
            return [
                [
                    1 / denominator1,
                    (-lambda_list[1] - lambda_list[2] - lambda_list[3]) / denominator1,
                    (
                        lambda_list[1] * lambda_list[2]
                        + lambda_list[1] * lambda_list[3]
                        + lambda_list[2] * lambda_list[3]
                    )
                    / denominator1,
                    (-lambda_list[1] * lambda_list[2] * lambda_list[3]) / denominator1,
                ],
                [
                    1 / denominator2,
                    (-lambda_list[0] - lambda_list[2] - lambda_list[3]) / denominator2,
                    (
                        lambda_list[0] * lambda_list[2]
                        + lambda_list[0] * lambda_list[3]
                        + lambda_list[2] * lambda_list[3]
                    )
                    / denominator2,
                    (-lambda_list[0] * lambda_list[2] * lambda_list[3]) / denominator2,
                ],
                [
                    1 / denominator3,
                    (-lambda_list[0] - lambda_list[1] - lambda_list[3]) / denominator3,
                    (
                        lambda_list[0] * lambda_list[1]
                        + lambda_list[0] * lambda_list[3]
                        + lambda_list[1] * lambda_list[3]
                    )
                    / denominator3,
                    (-lambda_list[0] * lambda_list[1] * lambda_list[3]) / denominator3,
                ],
                [
                    1 / denominator4,
                    (-lambda_list[0] - lambda_list[1] - lambda_list[2]) / denominator4,
                    (
                        lambda_list[0] * lambda_list[1]
                        + lambda_list[0] * lambda_list[2]
                        + lambda_list[1] * lambda_list[2]
                    )
                    / denominator4,
                    (-lambda_list[0] * lambda_list[1] * lambda_list[2]) / denominator4,
                ],
            ]

    def get_coefficients_fn(self, order, interval_start, interval_end, lambda_list, tau):
        assert order in [1, 2, 3, 4]
        assert order == len(lambda_list), "the length of lambda list must be equal to the order"
        coefficients = []
        lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        for i in range(order):
            coefficient = 0
            for j in range(order):
                if self.predict_x0:
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_positive(
                        order - 1 - j, interval_start, interval_end, tau
                    )
                else:
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_negative(
                        order - 1 - j, interval_start, interval_end
                    )
            coefficients.append(coefficient)
        assert len(coefficients) == order, "the length of coefficients does not match the order"
        return coefficients

    def stochastic_adams_bashforth_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor,
        noise: torch.Tensor,
        order: int,
        tau: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the SA-Predictor.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of SA-Predictor at this timestep.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if noise is None:
            if len(args) > 2:
                noise = args[2]
            else:
                raise ValueError(" missing `noise` as a required keyward argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing `order` as a required keyward argument")
        if tau is None:
            if len(args) > 4:
                tau = args[4]
            else:
                raise ValueError(" missing `tau` as a required keyward argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        gradient_part = torch.zeros_like(sample)
        h = lambda_t - lambda_s0
        lambda_list = []

        for i in range(order):
            si = self.step_index - i
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            lambda_list.append(lambda_si)

        gradient_coefficients = self.get_coefficients_fn(order, lambda_s0, lambda_t, lambda_list, tau)

        x = sample

        if self.predict_x0:
            if (
                order == 2
            ):  ## if order = 2 we do a modification that does not influence the convergence order similar to unipc. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                temp_sigma = self.sigmas[self.step_index - 1]
                temp_alpha_s, temp_sigma_s = self._sigma_to_alpha_sigma_t(temp_sigma)
                temp_lambda_s = torch.log(temp_alpha_s) - torch.log(temp_sigma_s)
                gradient_coefficients[0] += (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h**2 / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2))
                    / (lambda_s0 - temp_lambda_s)
                )
                gradient_coefficients[1] -= (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h**2 / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2))
                    / (lambda_s0 - temp_lambda_s)
                )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_output_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_output_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_s0) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_s0) * x + gradient_part + noise_part

        x_t = x_t.to(x.dtype)
        return x_t

    def stochastic_adams_moulton_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor,
        last_noise: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        tau: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the SA-Corrector.

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            this_timestep (`int`):
                The current timestep `t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The order of SA-Corrector at this step.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """

        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError(" missing`last_sample` as a required keyward argument")
        if last_noise is None:
            if len(args) > 2:
                last_noise = args[2]
            else:
                raise ValueError(" missing`last_noise` as a required keyward argument")
        if this_sample is None:
            if len(args) > 3:
                this_sample = args[3]
            else:
                raise ValueError(" missing`this_sample` as a required keyward argument")
        if order is None:
            if len(args) > 4:
                order = args[4]
            else:
                raise ValueError(" missing`order` as a required keyward argument")
        if tau is None:
            if len(args) > 5:
                tau = args[5]
            else:
                raise ValueError(" missing`tau` as a required keyward argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        gradient_part = torch.zeros_like(this_sample)
        h = lambda_t - lambda_s0
        lambda_list = []
        for i in range(order):
            si = self.step_index - i
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            lambda_list.append(lambda_si)

        model_prev_list = model_output_list + [this_model_output]

        gradient_coefficients = self.get_coefficients_fn(order, lambda_s0, lambda_t, lambda_list, tau)

        x = last_sample

        if self.predict_x0:
            if (
                order == 2
            ):  ## if order = 2 we do a modification that does not influence the convergence order similar to UniPC. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                gradient_coefficients[0] += (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2 * h))
                )
                gradient_coefficients[1] -= (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2 * h))
                )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_prev_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * last_noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * last_noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_s0) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_s0) * x + gradient_part + noise_part

        x_t = x_t.to(x.dtype)
        return x_t

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the SA-Solver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = self.step_index > 0 and self.last_sample is not None

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            current_tau = self.tau_func(self.timestep_list[-1])
            sample = self.stochastic_adams_moulton_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                last_noise=self.last_noise,
                this_sample=sample,
                order=self.this_corrector_order,
                tau=current_tau,
            )

        for i in range(max(self.config.predictor_order, self.config.corrector_order - 1) - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )

        if self.config.lower_order_final:
            this_predictor_order = min(self.config.predictor_order, len(self.timesteps) - self.step_index)
            this_corrector_order = min(self.config.corrector_order, len(self.timesteps) - self.step_index + 1)
        else:
            this_predictor_order = self.config.predictor_order
            this_corrector_order = self.config.corrector_order

        self.this_predictor_order = min(this_predictor_order, self.lower_order_nums + 1)  # warmup for multistep
        self.this_corrector_order = min(this_corrector_order, self.lower_order_nums + 2)  # warmup for multistep
        assert self.this_predictor_order > 0
        assert self.this_corrector_order > 0

        self.last_sample = sample
        self.last_noise = noise

        current_tau = self.tau_func(self.timestep_list[-1])
        prev_sample = self.stochastic_adams_bashforth_update(
            model_output=model_output_convert,
            sample=sample,
            noise=noise,
            order=self.this_predictor_order,
            tau=current_tau,
        )

        if self.lower_order_nums < max(self.config.predictor_order, self.config.corrector_order - 1):
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
