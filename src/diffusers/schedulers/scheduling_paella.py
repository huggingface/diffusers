# Copyright 2022 Microsoft and The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class PaellaSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
            Computed sample x_{t-1} of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.LongTensor


def log(t, eps=1e-20):
    # Compute the logarithm of t, with a small epsilon added to avoid log(0)
    return torch.log(t + eps)


def gumbel_noise(t):
    # Sample noise from a uniform distribution [0, 1]
    noise = torch.zeros_like(t).uniform_(0, 1)
    # Compute gumbel noise
    return -log(-log(noise))


def gumbel_sampling(t, temperature=1.0, dim=-1):
    # Sample from a categorical distribution using the Gumbel-Softmax trick
    # Divide t by the temperature and add gumbel noise
    # Then, take the argmax along the specified dimension to get the sample
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def locally_typical_sampling(latents_flat, typical_mass, typical_min_tokens):
    # Normalize the latents: transforms latents_flat into log probabilities
    latents_flat_norm = torch.nn.functional.log_softmax(latents_flat, dim=-1)
    # Calculates the probability distribution over labels by exponentiating the log probabilities obtained above
    latents_flat_norm_p = torch.exp(latents_flat_norm)
    # The entropy of the distribution is then calculated as the negative sum of the product of the probability distribution and the log probabilities.
    entropy = -(latents_flat_norm * latents_flat_norm_p).nansum(-1, keepdim=True)

    c_flat_shifted = torch.abs((-latents_flat_norm) - entropy)
    c_flat_sorted, latents_flat_indices = torch.sort(c_flat_shifted, descending=False)
    latents_flat_cumsum = latents_flat.gather(-1, latents_flat_indices).softmax(dim=-1).cumsum(dim=-1)

    last_ind = (latents_flat_cumsum < typical_mass).sum(dim=-1)
    sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
    if typical_min_tokens > 1:
        sorted_indices_to_remove[..., :typical_min_tokens] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, latents_flat_indices, sorted_indices_to_remove)
    # The final result of this function is a filtered version of the input tensor latents_flat.
    latents_flat = latents_flat.masked_fill(indices_to_remove, -float("Inf"))
    return latents_flat


class PaellaScheduler(SchedulerMixin, ConfigMixin):
    # TODO write documentation for PaellaScheduler
    """
    """

    # @register_to_config
    # def __init__(
    #     self,
    #     num_vec_classes: int,
    #     num_inference_steps: int = 12,
    # ):
    #     # TODO write __init__ for PaellaScheduler

    def set_temperatures(
        self,
        num_inference_steps: int,
        batch_size: int,
        temperature_range: Tuple[float, float] = [1.0, 1.0],
        device: Union[str, torch.device] = None,
    ):
        # TODO: write documentation for set_temperatures
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

            device (`str` or `torch.device`):
                device to place the timesteps and the diffusion process parameters (alpha, beta, gamma) on.
        """

        # Initialize  a sequence of temperatures temperatures that will be used to control the (Gumbel) sampling process.
        self.temperatures = torch.linspace(temperature_range[0], temperature_range[1], num_inference_steps)

        # rs is a tensor of evenly spaced values between 0 and 1 that are used to control the sampling process for the diffusion model
        # The value of r at each timestep is passed as an argument to the model's forward method, along with the current sample x and the class condition c.
        self.rs = torch.linspace(0, 1, num_inference_steps + 1)[:-1][:, None].expand(-1, batch_size).to(device)

    def step(
        self,
        idx,
        model_output,
        mask=None,
        temperature=1.0,
        do_locally_typical_sampling=True,
        typical_mass=0.2,
        typical_min_tokens=1,
        do_renoise=False,
        random_noise=None,
        start_latents=None,
    ):

        latents = model_output
        # Flatten the image to a 2D tensor of shape (batch_size * 32 * 32, num_vec_classes)
        latents_flat = latents.permute(0, 2, 3, 1).reshape(-1, latents.size(1))

        # If Locally Typical Sampling is enabled, apply it to the flattened image
        if do_locally_typical_sampling:
            latents_flat = locally_typical_sampling(latents_flat, typical_mass, typical_min_tokens)

        # After appling a softmax function to convert all scores to a probability distribution for each token in the latent image,
        # we sample one token from each distribution using using the Gumbel-Softmax trick and the current temperature.
        latents_flat = gumbel_sampling(latents_flat, temperature)
        latents = latents_flat.view(latents.size(0), *latents.shape[2:])

        if mask is not None:
            latents = latents * mask + (1 - mask) * start_latents

        if do_renoise:
            latents, _ = self.renoise(latents, self.rs[idx + 1], random_noise)

        return PaellaSchedulerOutput(prev_sample=latents)

    def renoise(self, x, r, random_noise=None):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(
            r * torch.ones_like(x),
        )
        mask = mask.round().long()
        if random_noise is None:
            random_noise = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_noise * mask
        return x, mask
