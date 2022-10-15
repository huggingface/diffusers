from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


'''
helper functions:   append_zero(),
                    get_sigmas(),
                    append_dims(),
                    get_scalings(),
                    DSsigma_to_t(),
                    DiscreteEpsDDPMDenoiserForward(),
                    
need cleaning 
'''



# def CFGDenoiserForward(Unet, x_in, sigma_in, cond_in,DSsigmas=None):
#         # x_in = torch.cat([x] * 2)#A# concat the latent
#         # sigma_in = torch.cat([sigma] * 2) #A# concat sigma
#         # cond_in = torch.cat([uncond, cond])
#         # uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
#         # uncond, cond = DiscreteEpsDDPMDenoiserForward(Unet,x_in, sigma_in,DSsigmas=DSsigmas, cond=cond_in).chunk(2)
#         # return uncond + (cond - uncond) * cond_scale
#         noise_pred = DiscreteEpsDDPMDenoiserForward(Unet,x_in, sigma_in,DSsigmas=DSsigmas, cond=cond_in)
#         return noise_pred


# def DiscreteEpsDDPMDenoiserForward(Unet,input,sigma,DSsigmas=None,**kwargs):
#     c_out, c_in = [append_dims(x, input.ndim) for x in get_scalings(sigma)]
#     #??? what is eps? 
#     # eps is the predicted added noise to the image Xt for noise level t
#     # eps = CVDget_eps(Unet,input * c_in, DSsigma_to_t(sigma), **kwargs)
#     eps = Unet(input * c_in, DSsigma_to_t(sigma,DSsigmas=DSsigmas), encoder_hidden_states=kwargs['cond']).sample
#     return input + eps * c_out




'''
Euler Ancestral Scheduler
'''

class EulerAScheduler(SchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

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
        tensor_format (`str`): whether the scheduler expects pytorch or numpy arrays.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        tensor_format: str = "pt",
        num_inference_steps = None,
        device = 'cuda'
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        # elif beta_schedule == "squaredcos_cap_v2":
        #     # Glide cosine schedule
        #     self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")


        self.device = device
        self.alphas = 1.0 - torch.from_numpy(self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # setable values
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        # get sigmas
        self.DSsigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = self.get_sigmas(self.DSsigmas,self.num_inference_steps)
        self.tensor_format = tensor_format
        
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas[0]
        
        self.set_format(tensor_format=tensor_format)
        
        
    #A# take number of steps as input
    #A# store 1) number of steps 2) timesteps 3) schedule 
    def set_timesteps(self, num_inference_steps: int, **kwargs):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        # offset = self.config.steps_offset

        # if "offset" in kwargs:
        #     warnings.warn(
        #         "`offset` is deprecated as an input argument to `set_timesteps` and will be removed in v0.4.0."
        #         " Please pass `steps_offset` to `__init__` instead.",
        #         DeprecationWarning,
        #     )

        #     offset = kwargs["offset"]

        # self.num_inference_steps = num_inference_steps
        # step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # # creates integer timesteps by multiplying by ratio
        # # casting to int to avoid issues when num_inference_step is power of 3
        # self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        # self.timesteps += offset
        self.timesteps = self.sigmas
        self.set_format(tensor_format=self.tensor_format)


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
        # c_out, c_in, sigma_in = self.prepare_input(sample, timestep)

        
        # noise_pred = latent_model_input + eps * c_out
        # sample * c_in, sigma_in
        # return sample *c_in
        c_out, c_in, sigma_in = self.prepare_input(sample, timestep)

        return sample * c_in

    def add_noise_to_input(
        self, sample: Union[torch.FloatTensor, np.ndarray], sigma: float, generator: Optional[torch.Generator] = None
    ) -> Tuple[Union[torch.FloatTensor, np.ndarray], float]:
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
        """
        if self.config.s_min <= sigma <= self.config.s_max:
            gamma = min(self.config.s_churn / self.num_inference_steps, 2**0.5 - 1)
        else:
            gamma = 0

        # sample eps ~ N(0, S_noise^2 * I)
        eps = self.config.s_noise * torch.randn(sample.shape, generator=generator).to(sample.device)
        sigma_hat = sigma + gamma * sigma
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)

        return sample_hat, sigma_hat

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        # timestep_prev: int,
        sample:float,
        generator: Optional[torch.Generator] = None,
        # ,sigma_hat: float,
        #  sigma_prev: float,
        # sample_hat: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor` or `np.ndarray`): TODO
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

            EulerAOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.EulerAOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.EulerAOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        
        latents = sample
        # ideally we could pass the index aka step to the this method
        # which will allow as to get the current timestep and the previous timestep 
        i = timestep # we are passing timestep as index 
        timestep = self.timesteps[i]
        prev_timestep = self.timesteps[i + 1]
        sigma_down, sigma_up = self.get_ancestral_step(timestep, prev_timestep)
        # if callback is not None:
        #     callback({'x': latents, 'i': i, 'sigma': timestep, 'sigma_hat': timestep, 'denoised': model_output})
        d = self.to_d(latents, timestep, model_output)
        # Euler method
        dt = sigma_down - timestep
        latents = latents + d * dt
        # latents = latents + self.randn_like(latents,generator=generator) * sigma_up # use self.randn_like instead of torch.randn_like to get deterministic output
        noise = torch.randn(latents.shape, dtype=latents.dtype, generator=generator).to(self.device)
        latents = latents + noise * sigma_up 
        return SchedulerOutput(prev_sample=latents)




    def step_correct(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        sigma_hat: float,
        sigma_prev: float,
        sample_hat: Union[torch.FloatTensor, np.ndarray],
        sample_prev: Union[torch.FloatTensor, np.ndarray],
        derivative: Union[torch.FloatTensor, np.ndarray],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. TODO complete description

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor` or `np.ndarray`): TODO
            sample_prev (`torch.FloatTensor` or `np.ndarray`): TODO
            derivative (`torch.FloatTensor` or `np.ndarray`): TODO
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
        pred_original_sample = sample_prev + sigma_prev * model_output
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        if not return_dict:
            return (sample_prev, derivative)

        return SchedulerOutput(prev_sample=sample_prev)

    def add_noise(self, original_samples, noise, timesteps):
        raise NotImplementedError()

    
    #from k_samplers sampling.py
    def get_ancestral_step(self, sigma_from, sigma_to):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up
    
    def t_to_sigma(self, t, sigmas):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        return (1 - w) * sigmas[low_idx] + w * sigmas[high_idx]


    def append_zero(self,x):
        return torch.cat([x, x.new_zeros([1])])


    def get_sigmas(self, sigmas, n=None):
        if n is None:
            return self.append_zero(sigmas.flip(0))
        t_max = len(sigmas) - 1 # = 999
        device = self.device
        t = torch.linspace(t_max, 0, n, device=device)
        # t = torch.linspace(t_max, 0, n, device=sigmas.device)
        return self.append_zero(self.t_to_sigma(t,sigmas))

    #from k_samplers utils.py
    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]

    # from k_samplers sampling.py
    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)
    

    def get_scalings(self, sigma):
        sigma_data = 1.
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        return c_out, c_in

    #DiscreteSchedule -> DS
    def DSsigma_to_t(self, sigma, quantize=None):
        # quantize = self.quantize if quantize is None else quantize
        quantize = False
        dists = torch.abs(sigma - self.DSsigmas[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigma.shape)
        low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
        low, high = self.DSsigmas[low_idx], self.DSsigmas[high_idx]
        w = (low - sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)
    
    def prepare_input(self,latent_in, t):
        sigma = t.reshape(1) 
        
        sigma_in = torch.cat([sigma] * latent_in.shape[0])# latent_in.shape[0] => 2 * batch_size 
        # noise_pred = CFGDenoiserForward(self.unet, latent_model_input, sigma_in, text_embeddings , guidance_scale,DSsigmas=self.scheduler.DSsigmas)
        # noise_pred = DiscreteEpsDDPMDenoiserForward(self.unet,latent_model_input, sigma_in,DSsigmas=self.scheduler.DSsigmas, cond=cond_in)
        c_out, c_in = [self.append_dims(x, latent_in.ndim) for x in self.get_scalings(sigma_in)]
        
        sigma_in = self.DSsigma_to_t(sigma_in)
        # s_in = latent_in.new_ones([latent_in.shape[0]])
        # sigma_in = sigma_in * s_in
        
        return c_out, c_in, sigma_in

    def get_sigma_in(self,latent_in, t):
        sigma = t.reshape(1)
        
        sigma_in = torch.cat([sigma] * latent_in.shape[0])# latent_in.shape[0] => 2 * batch_size 
        
        sigma_in = self.DSsigma_to_t(sigma_in)
        
        return sigma_in
    