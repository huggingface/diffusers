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

import numpy as np
import torch

import tqdm

from ...models.unet_1d import UNet1DModel
from ...pipeline_utils import DiffusionPipeline
from ...utils.dummy_pt_objects import DDPMScheduler


class ValueGuidedRLPipeline(DiffusionPipeline):
    def __init__(
        self,
        value_function: UNet1DModel,
        unet: UNet1DModel,
        scheduler: DDPMScheduler,
        env,
    ):
        super().__init__()
        self.value_function = value_function
        self.unet = unet
        self.scheduler = scheduler
        self.env = env
        self.data = env.get_dataset()
        self.means = dict()
        for key in self.data.keys():
            try:
                self.means[key] = self.data[key].mean()
            except:
                pass
        self.stds = dict()
        for key in self.data.keys():
            try:
                self.stds[key] = self.data[key].std()
            except:
                pass
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    def normalize(self, x_in, key):
        return (x_in - self.means[key]) / self.stds[key]

    def de_normalize(self, x_in, key):
        return x_in * self.stds[key] + self.means[key]

    def to_torch(self, x_in):
        if type(x_in) is dict:
            return {k: self.to_torch(v) for k, v in x_in.items()}
        elif torch.is_tensor(x_in):
            return x_in.to(self.unet.device)
        return torch.tensor(x_in, device=self.unet.device)

    def reset_x0(self, x_in, cond, act_dim):
        for key, val in cond.items():
            x_in[:, key, act_dim:] = val.clone()
        return x_in

    def run_diffusion(self, x, conditions, n_guide_steps, scale):
        batch_size = x.shape[0]
        y = None
        for i in tqdm.tqdm(self.scheduler.timesteps):
            # create batch of timesteps to pass into model
            timesteps = torch.full((batch_size,), i, device=self.unet.device, dtype=torch.long)
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    x.requires_grad_()
                    y = self.value_function(x.permute(0, 2, 1), timesteps).sample
                    grad = torch.autograd.grad([y.sum()], [x])[0]

                    posterior_variance = self.scheduler._get_variance(i)
                    model_std = torch.exp(0.5 * posterior_variance)
                    grad = model_std * grad
                grad[timesteps < 2] = 0
                x = x.detach()
                x = x + scale * grad
                x = self.reset_x0(x, conditions, self.action_dim)
            prev_x = self.unet(x.permute(0, 2, 1), timesteps).sample.permute(0, 2, 1)
            x = self.scheduler.step(prev_x, i, x, predict_epsilon=False)["prev_sample"]

            # apply conditions to the trajectory
            x = self.reset_x0(x, conditions, self.action_dim)
            x = self.to_torch(x)
        return x, y

    def __call__(self, obs, batch_size=64, planning_horizon=32, n_guide_steps=2, scale=0.1):
        # normalize the observations and create  batch dimension
        obs = self.normalize(obs, "observations")
        obs = obs[None].repeat(batch_size, axis=0)

        conditions = {0: self.to_torch(obs)}
        shape = (batch_size, planning_horizon, self.state_dim + self.action_dim)

        # generate initial noise and apply our conditions (to make the trajectories start at current state)
        x1 = torch.randn(shape, device=self.unet.device)
        x = self.reset_x0(x1, conditions, self.action_dim)
        x = self.to_torch(x)

        # run the diffusion process
        x, y = self.run_diffusion(x, conditions, n_guide_steps, scale)

        # sort output trajectories by value
        sorted_idx = y.argsort(0, descending=True).squeeze()
        sorted_values = x[sorted_idx]
        actions = sorted_values[:, :, : self.action_dim]
        actions = actions.detach().cpu().numpy()
        denorm_actions = self.de_normalize(actions, key="actions")

        # select the action with the highest value
        if y is not None:
            selected_index = 0
        else:
            # if we didn't run value guiding, select a random action
            selected_index = np.random.randint(0, batch_size)
        denorm_actions = denorm_actions[selected_index, 0]
        return denorm_actions
