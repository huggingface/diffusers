# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import inspect
import tempfile
import unittest

import numpy as np
import torch

from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    GlidePipeline,
    GlideSuperResUNetModel,
    GlideTextToImageUNetModel,
    LatentDiffusionPipeline,
    LatentDiffusionUncondPipeline,
    NCSNpp,
    PNDMPipeline,
    PNDMScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    ScoreSdeVpPipeline,
    ScoreSdeVpScheduler,
    UNetLDMModel,
    UNetModel,
    UNetUnconditionalModel,
    VQModel,
)
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.testing_utils import floats_tensor, slow, torch_device
from diffusers.training_utils import EMAModel


# 1. LDM

def test_output_pretrained_ldm_dummy():
    model = UNetUnconditionalModel.from_pretrained("fusing/unet-ldm-dummy", ldm=True)
    model.eval()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
    time_step = torch.tensor([10] * noise.shape[0])

    with torch.no_grad():
        output = model(noise, time_step)
        
    print(model)
    import ipdb; ipdb.set_trace()


def test_output_pretrained_ldm():
    model = UNetUnconditionalModel.from_pretrained("fusing/latent-diffusion-celeba-256", subfolder="unet", ldm=True)
    model.eval()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
    time_step = torch.tensor([10] * noise.shape[0])

    with torch.no_grad():
        output = model(noise, time_step)
            
    print(model)
    import ipdb; ipdb.set_trace()

# To see the how the final model should look like
# => this is the architecture in which the model should be saved in the new format
# -> verify new repo with the following tests (in `test_modeling_utils.py`)
# - test_ldm_uncond (in PipelineTesterMixin)
# - test_output_pretrained  ( in UNetLDMModelTests)

#test_output_pretrained_ldm_dummy()
#test_output_pretrained_ldm()


# 2. DDPM

def get_model(model_id):
    model = UNetUnconditionalModel.from_pretrained(model_id, ldm=True)

    noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
    time_step = torch.tensor([10] * noise.shape[0])

    with torch.no_grad():
        output = model(noise, time_step)

    print(model)

# Repos to convert and port to google (part of https://github.com/hojonathanho/diffusion)
# - fusing/ddpm_dummy
# - fusing/ddpm-cifar10
# - https://huggingface.co/fusing/ddpm-lsun-church-ema
# - https://huggingface.co/fusing/ddpm-lsun-bedroom-ema
# - https://huggingface.co/fusing/ddpm-celeba-hq

# tests to make sure to pass
# - test_ddim_cifar10, test_ddim_lsun, test_ddpm_cifar10, test_ddim_cifar10 (in PipelineTesterMixin)
# - test_output_pretrained  ( in UNetModelTests)

# e.g.
get_model("fusing/ddpm-cifar10")

# 3. NCSNpp

# Repos to convert and port to google (part of https://github.com/yang-song/score_sde)
# - https://huggingface.co/fusing/ffhq_ncsnpp
# - https://huggingface.co/fusing/church_256-ncsnpp-ve
# - https://huggingface.co/fusing/celebahq_256-ncsnpp-ve
# - https://huggingface.co/fusing/bedroom_256-ncsnpp-ve
# - https://huggingface.co/fusing/ffhq_256-ncsnpp-ve

# tests to make sure to pass
# - test_score_sde_ve_pipeline (in PipelineTesterMixin)
# - test_output_pretrained_ve_mid, test_output_pretrained_ve_large (in NCSNppModelTests)
