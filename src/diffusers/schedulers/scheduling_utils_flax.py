# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import importlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import flax
import jax.numpy as jnp

from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1
    FlaxDDPMScheduler = 2
    FlaxPNDMScheduler = 3
    FlaxLMSDiscreteScheduler = 4
    FlaxDPMSolverMultistepScheduler = 5


@dataclass
class FlaxSchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: jnp.ndarray


class FlaxSchedulerMixin:
    """
    Mixin containing common functions for the schedulers.

    Class attributes:
        - **_compatibles** (`List[str]`) -- A list of classes that are compatible with the parent class, so that
          `from_config` can be used from a class different than the one used to save the config (should be overridden
          by parent class).
    """

    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config = ["dtype"]
    _compatibles = []
    has_compatibles = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Dict[str, Any] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        r"""
        Instantiate a Scheduler class from a pre-defined JSON-file.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~SchedulerMixin.save_pretrained`],
                      e.g., `./my_model_directory/`.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        """
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            **kwargs,
        )
        scheduler, unused_kwargs = cls.from_config(config, return_unused_kwargs=True, **kwargs)

        if hasattr(scheduler, "create_state") and getattr(scheduler, "has_state", False):
            state = scheduler.create_state()

        if return_unused_kwargs:
            return scheduler, state, unused_kwargs

        return scheduler, state

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~FlaxSchedulerMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    @property
    def compatibles(self):
        """
        Returns all schedulers that are compatible with this scheduler

        Returns:
            `List[SchedulerMixin]`: List of compatible schedulers
        """
        return self._get_compatibles()

    @classmethod
    def _get_compatibles(cls):
        compatible_classes_str = list(set([cls.__name__] + cls._compatibles))
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        compatible_classes = [
            getattr(diffusers_library, c) for c in compatible_classes_str if hasattr(diffusers_library, c)
        ]
        return compatible_classes


def broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
    assert len(shape) >= x.ndim
    return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999, dtype=jnp.float32) -> jnp.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`jnp.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas, dtype=dtype)


@flax.struct.dataclass
class CommonSchedulerState:
    alphas: jnp.ndarray
    betas: jnp.ndarray
    alphas_cumprod: jnp.ndarray

    @classmethod
    def create(cls, scheduler):
        config = scheduler.config

        if config.trained_betas is not None:
            betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
        elif config.beta_schedule == "linear":
            betas = jnp.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=scheduler.dtype)
        elif config.beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                jnp.linspace(
                    config.beta_start**0.5, config.beta_end**0.5, config.num_train_timesteps, dtype=scheduler.dtype
                )
                ** 2
            )
        elif config.beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=scheduler.dtype)
        else:
            raise NotImplementedError(
                f"beta_schedule {config.beta_schedule} is not implemented for scheduler {scheduler.__class__.__name__}"
            )

        alphas = 1.0 - betas

        alphas_cumprod = jnp.cumprod(alphas, axis=0)

        return cls(
            alphas=alphas,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
        )


def get_sqrt_alpha_prod(
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    alphas_cumprod = state.alphas_cumprod

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(sqrt_one_minus_alpha_prod, original_samples.shape)

    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def add_noise_common(
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, original_samples, noise, timesteps)
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, sample, noise, timesteps)
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
