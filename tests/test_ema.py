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

import unittest

import gc
import numpy as np
import torch
from torch import nn

from diffusers.models.attention import GEGLU, AdaLayerNorm, ApproximateGELU, AttentionBlock
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers import UNet2DConditionModel
from diffusers.utils import torch_device
from diffusers.training_utils import EMAModel

class EMAModelTests(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"

    def get_models(self):
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        ema_unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        ema_unet = EMAModel(
            ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config
        )
        return unet, ema_unet 
    
    def test_optimization_steps_updated(self):
        unet, ema_unet = self.get_models()
        # Take the first (hypothetical) EMA step.
        ema_unet.step(unet.parameters())
        assert ema_unet.optimization == 1

        # Take two more.
        for _ in range(2):
            ema_unet.step(unet.parameters())
        assert ema_unet.optimization == 3

        del unet, ema_unet

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
        
        del unet, ema_unet

    def test_shadow_params_updated(self):
        unet, ema_unet = self.get_models()
        # Here we simulate the parameter updates for `unet`. Since there might
        # be some parameters which are initialized to zero we take extra care to
        # initialize their values to something non-zero before the multiplication.
        updated_params = []
        for param in unet.parameters(): 
            updated_params.append(torch.randn_like(param) + (param * torch.randn_like(param)))

        # Load the updated parameters into `unet`.
        updated_state_dict = {}
        for i, k in enumerate(unet.state_dict().keys()):
            updated_state_dict.update({k: updated_params[i]})
        unet.load_state_dict(updated_state_dict)

        # Take the EMA step.
        ema_unet.step(unet.parameters())

        # Now the EMA'd parameters won't be equal to the original model parameters.
        orig_params = list(unet.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert ~torch.allclose(s_param, param)

        # Ensure this is the case when we take multiple EMA steps.
        for _ in range(4):
            ema_unet.step(unet.parameters())
        for s_param, param in zip(ema_unet.shadow_params, orig_params):
            assert ~torch.allclose(s_param, param)

    def test_consecutive_shadow_params_not_updated(self):
        # EMA steps are supposed to be taken after we have taken a backprop step.
        # If that is not the case shadown params after two consecutive steps should
        # be one and the same
        pass

    def test_consecutive_shadow_params_updated(self):
        pass



    def tearDown(self):
        super().tearDown()
        gc.collect()
