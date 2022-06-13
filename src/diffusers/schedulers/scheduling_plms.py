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
from .schedulers_utils import SchedulerMixin, betas_for_alpha_bar, linear_beta_schedule


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_predicted_image=True,
        tensor_format="np",
    ):
        super().__init__()
        self.register(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )
        self.timesteps = int(timesteps)
        self.clip_image = clip_predicted_image

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
        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

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

    def get_orig_t(self, t, num_inference_steps):
        if t < 0:
            return -1
        return self.timesteps // num_inference_steps * t

    def get_variance(self, t, num_inference_steps):
        orig_t = self.get_orig_t(t, num_inference_steps)
        orig_prev_t = self.get_orig_t(t - 1, num_inference_steps)

        alpha_prod_t = self.get_alpha_prod(orig_t)
        alpha_prod_t_prev = self.get_alpha_prod(orig_prev_t)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(self, residual, image, t, num_inference_steps, eta):
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_image -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_image_direction -> "direction pointingc to x_t"
        # - pred_prev_image -> "x_t-1"

        # 1. get actual t and t-1
        orig_t = self.get_orig_t(t, num_inference_steps)
        orig_prev_t = self.get_orig_t(t - 1, num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.get_alpha_prod(orig_t)
        alpha_prod_t_prev = self.get_alpha_prod(orig_prev_t)
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original image from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_image = (image - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.clip_image:
            pred_original_image = self.clip(pred_original_image, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.get_variance(t, num_inference_steps)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_image_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * residual

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_prev_image = alpha_prod_t_prev ** (0.5) * pred_original_image + pred_image_direction

        return pred_prev_image

    def __len__(self):
        return self.timesteps
