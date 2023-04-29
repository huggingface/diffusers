# Copyright 2023 NVIDIA and The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from piq import LPIPS

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, randn_tensor
from .scheduling_utils import SchedulerMixin


@dataclass
class CMOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    #derivative: torch.FloatTensor
    #pred_original_sample: Optional[torch.FloatTensor] = None


class CMScheduler(SchedulerMixin, ConfigMixin):
    """
    The training and inference code of **CT** from consistency models [1].
    **CD** from consistency models has not been verified.
    The source code refers to https://github.com/openai/consistency_models
    
    [1] Song, Yang, et al. "Consistency Models." https://arxiv.org/abs/2303.01469 

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].

    """

    order = 2
    
    @register_to_config
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        s_noise: float = 1.0,
        s_churn: float = 0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        rho: float = 7.0,
        sampled_timesteps: list = [0,106,200],
        num_inference_steps: int = 201,
        target_ema_mode: str = "adaptive",
        start_ema: float = 0.95,
        scale_mode: str = "progressive",
        start_scales: int = 2,
        end_scales: int = 200,
        total_steps: int = 800000,
        weight_schedule: str = "uniform",
        loss_norm: str = "lpips",
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.num_inference_steps = num_inference_steps
        self.timesteps = sampled_timesteps
        self.schedule: torch.FloatTensor = None  # sigma(t_i)
        
        self.sigma_data = sigma_data
        self.sampled_timesteps = sampled_timesteps
        
        self.target_ema_mode = target_ema_mode
        self.start_ema = start_ema
        self.scale_mode = scale_mode
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.total_steps = total_steps
        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm
        
        self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(self.num_inference_steps, dtype=torch.float32, device=device)
        
        self.schedule = self.get_edm_timesteps(self.timesteps, num_inference_steps)
        
        self.timesteps = self.timesteps[self.config.sampled_timesteps]


    def step(
        self,
        model,
        sample,
        sigma=None,
        return_dict: bool = True,
    ) -> Union[CMOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than CMOutput class

            CMOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.CMOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.CMOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """

        ts = self.config.sampled_timesteps
        
        if sigma is None:
            #sigmas = self.get_sigmas_karras(n=1, device=sample.device)
            sigmas = self.schedule[ts[0]]
        else:
            sigmas = sigma
        
        sample_prev = self.denoise(sample, sigmas, model)

        if not return_dict:
            return (sample_prev, )

        return CMOutput(
            prev_sample=sample_prev 
        )
    
    def denoise(self, x_t, sigmas, model):
        c_skip, c_out, c_in = [
            x[(...,) + (None,) * ((x_t.ndim - x.ndim) * (x_t.ndim - x.ndim > 0))]
            for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t).sample
        sample_prev = c_out * model_output + c_skip * x_t
        return sample_prev
    
    #def get_sigmas_karras(self, n=None, device="cpu"):
    #    """Constructs the noise schedule of Karras et al. (2022)."""
    #    if n is None:
    #        n = self.config.num_inference_steps
    #    
    #    ramp = torch.linspace(0, 1, n).to(device) * (n-1)
    #    
    #    sigmas = self.get_edm_timesteps(ramp, n)
    #    
    #    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    #    return sigmas

    def get_edm_timesteps(self, ts_indices, num_scales, is_clip=False):
        t_max = self.config.sigma_max
        t_min = self.config.sigma_min
        rho = self.config.rho
        
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        
        t = (t_max_rho + ts_indices * 1. / (num_scales - 1) * (t_min_rho - t_max_rho)) ** rho
        
        if is_clip:
            t = t.clamp(min=t_min, max=t_max)
        
        return t

    def add_noise_to_input(
        self, 
        sample: torch.FloatTensor, 
        generator: Optional[torch.Generator] = None,
        i: int = 0,
    ) -> Tuple[torch.FloatTensor, float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
        """
        t_max = self.config.sigma_max
        t_min = self.config.sigma_min
        
        ts = self.config.sampled_timesteps
        
        #steps = self.config.num_inference_steps
        #t = self.get_edm_timesteps(ts[i], steps)
        #next_t = self.get_edm_timesteps(ts[i+1], steps, is_clip=True)
        t = self.schedule[ts[i]]
        next_t = self.schedule[ts[i+1]].clamp(min=t_min, max=t_max)
        sigma = t
        sigma_hat = next_t

        # sample z ~ N(0, S_noise^2 * I)
        z = self.config.s_noise * randn_tensor(sample.shape, generator=generator, device=sample.device)
        
        # tau = sigma_hat, eps = t_min
        sample_hat = sample + ((sigma_hat**2 - t_min**2) ** 0.5 * z)

        return sample_hat, sigma_hat, sigma

    def add_noise(self, original_samples, global_step, bsz):
        # adding noise for training
        noise = torch.randn_like(original_samples)

        target_ema, num_scales = self.create_ema_and_scales_fn(global_step)
        
        timesteps = torch.randint(
            0, num_scales-1, (bsz,), device=original_samples.device
        ).long()

        t_i = self.get_edm_timesteps(timesteps, num_scales)

        sigma = t_i.flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + sigma * noise
        return noisy_samples, timesteps, num_scales

    def create_ema_and_scales_fn(
        self,
        step,
    ):
        target_ema_mode = self.target_ema_mode 
        start_ema = self.start_ema 
        scale_mode = self.scale_mode 
        start_scales = self.start_scales 
        end_scales = self.end_scales 
        total_steps = self.total_steps 

        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1
    
        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        else:
            raise NotImplementedError
    
        return float(target_ema), int(scales)


    def get_scalings(self, sigma):
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (
            (sigma - self.config.sigma_min) ** 2 + sigma_data**2
        )
        c_out = (
            (sigma - self.config.sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
    
    
    def consistency_loss(
        self, 
        clean_images, 
        noisy_images, 
        num_scales, 
        timesteps, 
        model, 
        target_model, 
        teacher_model=None,
    ):
        t = self.get_edm_timesteps(timesteps, num_scales)
        t2 = self.get_edm_timesteps(timesteps+1, num_scales)
        
        dropout_state = torch.get_rng_state()
        
        model_output = self.denoise(noisy_images, t, model)
        distiller = model_output
        x_t = noisy_images
        x_start = clean_images
        
        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()
        
        torch.set_rng_state(dropout_state)
        
        #target_model.store(tmp_target_model.parameters())
        #target_model.copy_to(tmp_target_model.parameters())
        
        #distiller_target = self.denoise(x_t2, t2, tmp_target_model)
        distiller_target = self.denoise(x_t2, t2, target_model)
        distiller_target = distiller_target.detach()
        
        snrs = self.get_snr(t)
        weights = self.get_weightings(self.weight_schedule, snrs, self.config.sigma_data)
        
        if self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        
        #target_model.restore(tmp_target_model.parameters())
        
        return loss
    
    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_weightings(self, weight_schedule, snrs, sigma_data):
        if weight_schedule == "snr":
            weightings = snrs
        elif weight_schedule == "snr+1":
            weightings = snrs + 1
        elif weight_schedule == "karras":
            weightings = snrs + 1.0 / sigma_data**2
        elif weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()
        return weightings


@torch.no_grad()
def heun_solver(samples, t, next_t, x0, teacher_model=None):
    dims = x0.ndim
    x = samples
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_model(x, t)

    d = (x - denoiser) / append_dims(t, dims)
    samples = x + d * append_dims(next_t - t, dims)
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_model(samples, next_t)

    next_d = (samples - denoiser) / append_dims(next_t, dims)
    samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

    return samples

@torch.no_grad()
def euler_solver(samples, t, next_t, x0, teacher_model=None):
    dims = x0.ndim
    x = samples
    if teacher_model is None:
        denoiser = x0
    else:
        denoiser = teacher_model(x, t)
    d = (x - denoiser) / append_dims(t, dims)
    samples = x + d * append_dims(next_t - t, dims)

    return samples

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))