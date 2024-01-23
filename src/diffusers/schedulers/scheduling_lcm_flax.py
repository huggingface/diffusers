from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    add_noise_common,
    get_velocity_common,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def rescale_zero_terminal_snr(betas: jnp.ndarray) -> jnp.ndarray:
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = jnp.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

@flax.struct.dataclass
class LCMSchedulerState:
    common: CommonSchedulerState
    final_alpha_cumprod: jnp.ndarray

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None
    custom_timesteps: Optional[bool] = False
    step_index: Optional[int] = -1 

    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        final_alpha_cumprod: jnp.ndarray,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
        custom_timesteps: bool = False,
        step_index: int = None
    ):
        return cls(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            custom_timesteps=custom_timesteps,
            step_index=step_index
        )
        

@dataclass
class FlaxLCMSchedulerOutput(FlaxSchedulerOutput):
    state: LCMSchedulerState


class FlaxLCMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    `LCMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. [`~ConfigMixin`] takes care of storing all config
    attributes that are passed in the scheduler's `__init__` function, such as `num_train_timesteps`. They can be
    accessed via `scheduler.config.num_train_timesteps`. [`SchedulerMixin`] provides general loading and saving
    functionality via the [`SchedulerMixin.save_pretrained`] and [`~SchedulerMixin.from_pretrained`] functions.

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
        original_inference_steps (`int`, *optional*, defaults to 50):
            The default number of inference steps used to generate a linearly-spaced timestep schedule, from which we
            will ultimately take `num_inference_steps` evenly spaced timesteps to form the final timestep schedule.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product like in Stable
            Diffusion.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        timestep_scaling (`float`, defaults to 10.0):
            The factor the timesteps will be multiplied by when calculating the consistency model boundary conditions
            `c_skip` and `c_out`. Increasing this will decrease the approximation error (although the approximation
            error at the default of `10.0` is already pretty small).
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    order = 1
    
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    dtype: jnp.dtype

    @property
    def has_state(self):
        return True

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear", 
        trained_betas: Optional[Union[jnp.ndarray, List[float]]] = None,
        original_inference_steps: int = 50,  # LCM scheduler
        clip_sample: bool = False,           # LCM scheduler
        clip_sample_range: float = 1.0,      # LCM scheduler
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,                # LCM scheduler
        dynamic_thresholding_ratio: float = 0.995, # LCM scheduler
        sample_max_value: float = 1.0,             # LCM scheduler
        timestep_spacing: str = "leading",         # LCM scheduler
        timestep_scaling: float = 10.0,            # LCM scheduler
        rescale_betas_zero_snr: bool = False,      # LCM scheduler
        dtype: jnp.dtype = jnp.float32,

    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.original_inference_steps = original_inference_steps
        self.dtype = dtype

    def create_state(self, common: Optional[CommonSchedulerState] = None) -> LCMSchedulerState:
        if common is None:
            common = CommonSchedulerState.create(self)
        
        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        final_alpha_cumprod = (
            jnp.array(1.0, dtype=self.dtype) if self.config.set_alpha_to_one else common.alphas_cumprod[0]
        )
        
        # Rescale for zero SNR
        if self.config.rescale_betas_zero_snr:
            common.betas = rescale_zero_terminal_snr(common.betas)

        # standard deviation of the initial noise distribution
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        custom_timesteps = False
        step_index = -1

        return LCMSchedulerState.create(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            custom_timesteps=custom_timesteps,
            step_index=step_index
        )
    
    def _init_step_index(self, state, timestep):
        (step_index,) = jnp.where(state.timesteps == timestep, size=2)
        step_index = jax.lax.select(len(step_index) > 1, step_index[1], step_index[0])
        return step_index
   
    def scale_model_input(self, state: LCMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None) -> jnp.ndarray:
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
        return sample
    
    def set_timesteps(
        self, 
        state: LCMSchedulerState,
        shape: Tuple = (),
        num_inference_steps: Optional[int] = None,
        original_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        strength: int = 1.0) -> LCMSchedulerState:
    

        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DDIMSchedulerState`):
                the `FlaxDDIMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the LCM original training/distillation timestep schedule.
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.original_inference_steps
        )

        if original_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # LCM Timesteps Setting
        # The skipping step parameter k from the paper
        k = self.config.num_train_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = jnp.array(list(range(1, int(original_steps * strength) + 1))) * k - 1

        skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        
        if skipping_step < 1:
            raise ValueError(
                f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
            )
    
        if num_inference_steps > original_steps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                f" {original_steps} because the final timestep schedule will be a subset of the"
                f" `original_inference_steps`-sized initial timestep schedule."
            )

        # LCM Inference Steps Schedule
        lcm_origin_timesteps = lcm_origin_timesteps[::-1]
        inference_indices = jnp.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = jnp.floor(inference_indices).astype(jnp.int32)
        timesteps = lcm_origin_timesteps[jnp.array(inference_indices)]
        step_ratio = self.config.num_train_timesteps // num_inference_steps

        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            step_index=-1
        )

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5  # Default: 0.5

        scaled_timestep = timestep * self.config.timestep_scaling

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        state: LCMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: Optional[jax.Array] = None,
        return_dict: bool = True,
    ) -> Union[FlaxLCMSchedulerOutput, Tuple]:
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
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        step_index = jax.lax.select(state.step_index == -1, self._init_step_index(state, timestep), state.step_index)
        
        # 1. get previous step value        
        prev_step_index = step_index + 1
        prev_timestep = jax.lax.select(prev_step_index < len(state.timesteps), state.timesteps[prev_step_index], timestep)
        # 2. compute alphas, betas
        alpha_prod_t = state.common.alphas_cumprod[timestep]
        alpha_prod_t_prev = jax.lax.select(prev_timestep >=0, state.common.alphas_cumprod[prev_timestep], state.final_alpha_cumprod)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.config.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - jnp.sqrt(beta_prod_t) * model_output) / jnp.sqrt(alpha_prod_t)
        elif self.config.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = jnp.sqrt(alpha_prod_t) * sample - jnp.sqrt(beta_prod_t) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction` for `LCMScheduler`."
            )

        # 5. Clip or threshold "predicted x_0"
        # if self.config.thresholding:
        #     predicted_original_sample = self._threshold_sample(predicted_original_sample)
        # elif self.config.clip_sample:
        #     predicted_original_sample = predicted_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )

        # 6. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.

        def get_noise(key, shape, dtype):
            return jax.random.normal(key, shape=shape, dtype=dtype)

        prev_sample = jax.lax.select(step_index != state.num_inference_steps - 1, 
                                     jnp.sqrt(alpha_prod_t_prev) * denoised + jnp.sqrt(beta_prod_t_prev) * get_noise(key, model_output.shape, denoised.dtype) , 
                                     denoised)
        # upon completion increase step index by one
        state = state.replace(
            step_index=step_index + 1
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxLCMSchedulerOutput(prev_sample=prev_sample, state=state)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        state: LCMSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return add_noise_common(state.common, original_samples, noise, timesteps)


    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
    def get_velocity(
        self, state: LCMSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
    ) -> jnp.ndarray:
        return get_velocity_common(state.common, sample, noise, timesteps)


    def __len__(self):
        return self.config.num_train_timesteps