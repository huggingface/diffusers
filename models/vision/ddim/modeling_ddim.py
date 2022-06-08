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


from diffusers import DiffusionPipeline
import tqdm
import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


class DDIM(DiffusionPipeline):

    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.register_modules(unet=unet, noise_scheduler=noise_scheduler)

    def __call__(self, batch_size=1, generator=None, torch_device=None, eta=0.0, inference_time_steps=50):
        # eta is η in paper

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        num_timesteps = self.noise_scheduler.num_timesteps

        seq = range(0, num_timesteps, num_timesteps // inference_time_steps)
        b = self.noise_scheduler.betas.to(torch_device)

        self.unet.to(torch_device)
        x = self.noise_scheduler.sample_noise((batch_size, self.unet.in_channels, self.unet.resolution, self.unet.resolution), device=torch_device, generator=generator)

        with torch.no_grad():
            n = batch_size
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                print(i)
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                et = self.unet(xt, t)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                # eta
                c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))

        return xt_next
