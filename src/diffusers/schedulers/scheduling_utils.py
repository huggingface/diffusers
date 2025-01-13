# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub.utils import validate_hf_hub_args

from ..utils import BaseOutput, PushToHubMixin
from .schedules.beta_schedule import BetaSchedule
from .schedules.flow_schedule import FlowMatchSchedule
from .sigmas.beta_sigmas import BetaSigmas
from .sigmas.exponential_sigmas import ExponentialSigmas
from .sigmas.karras_sigmas import KarrasSigmas


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


# NOTE: We make this type an enum because it simplifies usage in docs and prevents
# circular imports when used for `_compatibles` within the schedulers module.
# When it's used as a type in pipelines, it really is a Union because the actual
# scheduler instance is passed in.
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15


AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}

SCHEDULE_MAP = {
    "BetaSchedule": BetaSchedule,
    "FlowMatchSchedule": FlowMatchSchedule,
}

SIGMA_SCHEDULE_MAP = {
    "BetaSigmas": BetaSigmas,
    "ExponentialSigmas": ExponentialSigmas,
    "KarrasSigmas": KarrasSigmas,
}


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the output of a scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class SchedulerMixin(PushToHubMixin):
    """
    Base class for all schedulers.

    [`SchedulerMixin`] contains common functions shared by all schedulers such as general loading and saving
    functionalities.

    [`ConfigMixin`] takes care of storing the configuration attributes (like `num_train_timesteps`) that are passed to
    the scheduler's `__init__` function, and the attributes can be accessed by `scheduler.config.num_train_timesteps`.

    Class attributes:
        - **_compatibles** (`List[str]`) -- A list of scheduler classes that are compatible with the parent scheduler
          class. Use [`~ConfigMixin.from_config`] to load a different compatible scheduler class (should be overridden
          by parent class).
    """

    config_name = SCHEDULER_CONFIG_NAME
    _compatibles = []
    has_compatibles = True
    schedule_configs = SCHEDULE_MAP
    sigma_configs = SIGMA_SCHEDULE_MAP
    _schedule = None
    _sigma_schedule = None

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        r"""
        Instantiate a scheduler from a pre-defined JSON configuration file in a local directory or Hub repository.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the scheduler
                      configuration saved with [`~SchedulerMixin.save_pretrained`].
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        """
        config, kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a scheduler configuration object to a directory so that it can be reloaded using the
        [`~SchedulerMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
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

    def set_schedule(self, schedule: Union[Dict]):
        if isinstance(schedule, dict):
            class_name = schedule.get("class_name", None)
            if class_name is None:
                raise ValueError("Schedule config `class_name` is None.")
            elif class_name not in self.schedule_configs:
                raise ValueError(f"Expected one of {self.schedule_configs.keys()}")
            _class = self.schedule_configs[class_name]
            self._schedule = _class(**schedule)
        else:
            self._schedule = schedule

    def set_sigma_schedule(self, sigma_schedule: Union[Dict]):
        if isinstance(sigma_schedule, dict):
            if not sigma_schedule:
                self._sigma_schedule = None
                return
            class_name = sigma_schedule.get("class_name", None)
            if class_name is None:
                raise ValueError("Schedule config `class_name` is None.")
            elif class_name not in self.sigma_configs:
                raise ValueError(f"Expected one of {self.sigma_configs.keys()}")
            _class = self.sigma_configs[class_name]
            sigma_min = getattr(self._schedule, "sigma_min", None) or sigma_schedule.get("sigma_min", None)
            sigma_max = getattr(self._schedule, "sigma_max", None) or sigma_schedule.get("sigma_max", None)
            sigma_schedule.update(
                {
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                }
            )
            self._sigma_schedule = _class(**sigma_schedule)
        else:
            self._sigma_schedule = sigma_schedule


class SamplingMixin:
    _step_index = None
    _begin_index = None
    timesteps = None
    num_inference_steps = None
    is_scale_input_called = False

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

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        if self._schedule.__class__.__name__ == "FlowMatchSchedule":
            noisy_samples = (1.0 - sigma) * original_samples + noise * sigma
        else:
            noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self._schedule.__class__.__name__ != "BetaSchedule":
            raise ValueError("`get_velocity` only supports `BetaSchedule`.")
        if (
            isinstance(timesteps, int)
            or isinstance(timesteps, torch.IntTensor)
            or isinstance(timesteps, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if sample.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timesteps = timesteps.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timesteps = timesteps.to(sample.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        alphas_cumprod = self._schedule.alphas_cumprod.to(sample)
        sqrt_alpha_prod = alphas_cumprod[step_indices] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[step_indices]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

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

    def __len__(self):
        return self.config.num_train_timesteps
