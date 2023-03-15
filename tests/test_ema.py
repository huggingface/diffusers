# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import tempfile
import unittest

import torch

from diffusers import UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.utils.testing_utils import skip_mps, torch_device


class EMAModelTests(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    batch_size = 1
    prompt_length = 77
    text_encoder_hidden_dim = 32
    num_in_channels = 4
    latent_height = latent_width = 64
    generator = torch.manual_seed(0)

    def get_models(self, decay=0.9999):
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        unet = unet.to(torch_device)
        ema_unet = EMAModel(unet.parameters(), decay=decay, model_cls=UNet2DConditionModel, model_config=unet.config)
        return unet, ema_unet

    def get_dummy_inputs(self):
        noisy_latents = torch.randn(
            self.batch_size, self.num_in_channels, self.latent_height, self.latent_width, generator=self.generator
        ).to(torch_device)
        timesteps = torch.randint(0, 1000, size=(self.batch_size,), generator=self.generator).to(torch_device)
        encoder_hidden_states = torch.randn(
            self.batch_size, self.prompt_length, self.text_encoder_hidden_dim, generator=self.generator
        ).to(torch_device)
        return noisy_latents, timesteps, encoder_hidden_states

    def simulate_backprop(self, unet):
        updated_state_dict = {}
        for k, param in unet.state_dict().items():
            updated_param = torch.randn_like(param) + (param * torch.randn_like(param))
            updated_state_dict.update({k: updated_param})
        unet.load_state_dict(updated_state_dict)
        return unet

    def test_optimization_steps_updated(self):
        unet, ema_unet = self.get_models()
        # Take the first (hypothetical) EMA step.
        ema_unet.step(unet.parameters())
        assert ema_unet.optimization_step == 1

        # Take two more.
        for _ in range(2):
            ema_unet.step(unet.parameters())
        assert ema_unet.optimization_step == 3

    def test_shadow_params_not_updated(self):
        unet, ema_unet = self.get_models()
        # Since the `unet` is not being updated (i.e., backprop'd)
        # there won't be any difference between the `params` of `unet`
        # and `ema_unet` even if we call `ema_unet.step(unet.parameters())`.
        ema_unet.step(unet.parameters())
        orig_params = list(unet.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert torch.allclose(s_param, param)

        # The above holds true even if we call `ema.step()` multiple times since
        # `unet` params are still not being updated.
        for _ in range(4):
            ema_unet.step(unet.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert torch.allclose(s_param, param)

    def test_shadow_params_updated(self):
        unet, ema_unet = self.get_models()
        # Here we simulate the parameter updates for `unet`. Since there might
        # be some parameters which are initialized to zero we take extra care to
        # initialize their values to something non-zero before the multiplication.
        unet_pseudo_updated_step_one = self.simulate_backprop(unet)

        # Take the EMA step.
        ema_unet.step(unet_pseudo_updated_step_one.parameters())

        # Now the EMA'd parameters won't be equal to the original model parameters.
        orig_params = list(unet_pseudo_updated_step_one.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert ~torch.allclose(s_param, param)

        # Ensure this is the case when we take multiple EMA steps.
        for _ in range(4):
            ema_unet.step(unet.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert ~torch.allclose(s_param, param)

    def test_consecutive_shadow_params_updated(self):
        # If we call EMA step after a backpropagation consecutively for two times,
        # the shadow params from those two steps should be different.
        unet, ema_unet = self.get_models()

        # First backprop + EMA
        unet_step_one = self.simulate_backprop(unet)
        ema_unet.step(unet_step_one.parameters())
        step_one_shadow_params = ema_unet.shadow_params

        # Second backprop + EMA
        unet_step_two = self.simulate_backprop(unet_step_one)
        ema_unet.step(unet_step_two.parameters())
        step_two_shadow_params = ema_unet.shadow_params

        for step_one, step_two in zip(step_one_shadow_params, step_two_shadow_params):
            assert ~torch.allclose(step_one, step_two)

    def test_zero_decay(self):
        # If there's no decay even if there are backprops, EMA steps
        # won't take any effect i.e., the shadow params would remain the
        # same.
        unet, ema_unet = self.get_models(decay=0.0)
        unet_step_one = self.simulate_backprop(unet)
        ema_unet.step(unet_step_one.parameters())
        step_one_shadow_params = ema_unet.shadow_params

        unet_step_two = self.simulate_backprop(unet_step_one)
        ema_unet.step(unet_step_two.parameters())
        step_two_shadow_params = ema_unet.shadow_params

        for step_one, step_two in zip(step_one_shadow_params, step_two_shadow_params):
            assert torch.allclose(step_one, step_two)

    @skip_mps
    def test_serialization(self):
        unet, ema_unet = self.get_models()
        noisy_latents, timesteps, encoder_hidden_states = self.get_dummy_inputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            ema_unet.save_pretrained(tmpdir)
            loaded_unet = UNet2DConditionModel.from_pretrained(tmpdir, model_cls=UNet2DConditionModel)
            loaded_unet = loaded_unet.to(unet.device)

        # Since no EMA step has been performed the outputs should match.
        output = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        output_loaded = loaded_unet(noisy_latents, timesteps, encoder_hidden_states).sample

        assert torch.allclose(output, output_loaded)
