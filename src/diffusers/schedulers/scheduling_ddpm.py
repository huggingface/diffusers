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
import math

import numpy as np

from ..configuration_utils import ConfigMixin
from .scheduling_utils import SchedulerMixin, betas_for_alpha_bar, linear_beta_schedule


class DDPMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
        clip_predicted_image=True,
        tensor_format="np",
    ):
        super().__init__()
        self.register(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_predicted_image=clip_predicted_image,
        )
        self.timesteps = int(timesteps)
        self.clip_image = clip_predicted_image
        self.variance_type = variance_type

        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "squaredcos_cap_v2":
            # GLIDE cosine schedule
            self.betas = betas_for_alpha_bar(
                timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)
        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

    #        self.register_buffer("betas", betas.to(torch.float32))
    #        self.register_buffer("alphas", alphas.to(torch.float32))
    #        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))

    #        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    # TODO(PVP) - check how much of these is actually necessary!
    # LDM only uses "fixed_small"; glide seems to use a weird mix of the two, ...
    # https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/gaussian_diffusion.py#L246
    #        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    #        if variance_type == "fixed_small":
    #            log_variance = torch.log(variance.clamp(min=1e-20))
    #        elif variance_type == "fixed_large":
    #            log_variance = torch.log(torch.cat([variance[1:2], betas[1:]], dim=0))
    #
    #
    #        self.register_buffer("log_variance", log_variance.to(torch.float32))

    def get_alpha(self, time_step):
        return self.alphas[time_step]

    def get_beta(self, time_step):
        return self.betas[time_step]

    def get_alpha_prod(self, time_step):
        if time_step < 0:
            return self.one
        return self.alphas_cumprod[time_step]

    def get_variance(self, t):
        alpha_prod_t = self.get_alpha_prod(t)
        alpha_prod_t_prev = self.get_alpha_prod(t - 1)

        # For t > 0, compute predicted variance βt (see formala (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous image
        # x_{t-1} ~ N(pred_prev_image, variance) == add variane to pred_image
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.get_beta(t)

        # hacks - were probs added for training stability
        if self.variance_type == "fixed_small":
            variance = self.clip(variance, min_value=1e-20)
        elif self.variance_type == "fixed_large":
            variance = self.get_beta(t)

        return variance

    def step(self, residual, image, t):
        # 1. compute alphas, betas
        alpha_prod_t = self.get_alpha_prod(t)
        alpha_prod_t_prev = self.get_alpha_prod(t - 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original image from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_image = (image - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)

        # 3. Clip "predicted x_0"
        if self.clip_predicted_image:
            pred_original_image = self.clip(pred_original_image, -1, 1)

        # 4. Compute coefficients for pred_original_image x_0 and current image x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_image_coeff = (alpha_prod_t_prev ** (0.5) * self.get_beta(t)) / beta_prod_t
        current_image_coeff = self.get_alpha(t) ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous image µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_image = pred_original_image_coeff * pred_original_image + current_image_coeff * image

        return pred_prev_image

    def forward_step(self, original_image, noise, t):
        noisy_image = self.sqrt_alphas_cumprod[t] * original_image + self.sqrt_one_minus_alphas_cumprod[t] * noise
        return noisy_image

    def __len__(self):
        return self.timesteps
