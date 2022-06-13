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


class PNDMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
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

    def step(self, img, t_start, t_end, model, ets):
#        img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)
#def gen_order_4(img, t, t_next, model, alphas_cump, ets):
        t_next, t = t_start, t_end

        noise_ = model(img.to("cuda"), t.to("cuda"))
        noise_ = noise_.to("cpu")

        t_list = [t, (t+t_next)/2, t_next]
        if len(ets) > 2:
            ets.append(noise_)
            noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
        else:
            noise = self.runge_kutta(img, t_list, model, ets, noise_)

        img_next = self.transfer(img.to("cpu"), t, t_next, noise)
        return img_next, ets

    def runge_kutta(self, x, t_list, model, ets, noise_):
        model = model.to("cuda")
        x = x.to("cpu")

        e_1 = noise_
        ets.append(e_1)
        x_2 = self.transfer(x, t_list[0], t_list[1], e_1)

        e_2 = model(x_2.to("cuda"), t_list[1].to("cuda"))
        e_2 = e_2.to("cpu")
        x_3 = self.transfer(x, t_list[0], t_list[1], e_2)

        e_3 = model(x_3.to("cuda"), t_list[1].to("cuda"))
        e_3 = e_3.to("cpu")
        x_4 = self.transfer(x, t_list[0], t_list[2], e_3)

        e_4 = model(x_4.to("cuda"), t_list[2].to("cuda"))
        e_4 = e_4.to("cpu")

        et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

        return et

    def transfer(self, x, t, t_next, et):
        alphas_cump = self.alphas_cumprod
        at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
        at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

        x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - 1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

        x_next = x + x_delta
        return x_next

    def __len__(self):
        return self.timesteps
