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


import tempfile
import unittest

import torch

from diffusers import (
    BDDM,
    DDIM,
    DDPM,
    GLIDE,
    PNDM,
    DDIMScheduler,
    DDPMScheduler,
    LatentDiffusion,
    PNDMScheduler,
    UNetModel,
)
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pipeline_bddm import DiffWave
from diffusers.testing_utils import floats_tensor, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class ConfigTester(unittest.TestCase):
    def test_load_not_from_mixin(self):
        with self.assertRaises(ValueError):
            ConfigMixin.from_config("dummy_path")

    def test_save_load(self):
        class SampleObject(ConfigMixin):
            config_name = "config.json"

            def __init__(
                self,
                a=2,
                b=5,
                c=(2, 5),
                d="for diffusion",
                e=[1, 3],
            ):
                self.register_to_config(a=a, b=b, c=c, d=d, e=e)

        obj = SampleObject()
        config = obj.config

        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)
            new_obj = SampleObject.from_config(tmpdirname)
            new_config = new_obj.config

        # unfreeze configs
        config = dict(config)
        new_config = dict(new_config)

        assert config.pop("c") == (2, 5)  # instantiated as tuple
        assert new_config.pop("c") == [2, 5]  # saved & loaded as list because of json
        assert config == new_config


class ModelTesterMixin(unittest.TestCase):
    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return (noise, time_step)

    def test_from_pretrained_save_pretrained(self):
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
        model.to(torch_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = UNetModel.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        dummy_input = self.dummy_input

        image = model(*dummy_input)
        new_image = new_model(*dummy_input)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model = UNetModel.from_pretrained("fusing/ddpm_dummy")
        model.to(torch_device)

        image = model(*self.dummy_input)

        assert image is not None, "Make sure output is not None"
    
    def test_save_load(self):
        pass
    
    def test_determinism(self):
        pass
    
    def test_output(self):
        pass
    
    def test_forward_signature(self):
        pass
    
    def test_model_from_config(self):
        pass

    def test_training(self):
        pass


class PipelineTesterMixin(unittest.TestCase):
    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
        schedular = DDPMScheduler(timesteps=10)

        ddpm = DDPM(model, schedular)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPM.from_pretrained(tmpdirname)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "fusing/ddpm-cifar10"

        ddpm = DDPM.from_pretrained(model_path)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path)

        ddpm.noise_scheduler.num_timesteps = 10
        ddpm_from_hub.noise_scheduler.num_timesteps = 10

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_ddpm_cifar10(self):
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = DDPMScheduler.from_config(model_id)
        noise_scheduler = noise_scheduler.set_format("pt")

        ddpm = DDPM(unet=unet, noise_scheduler=noise_scheduler)
        image = ddpm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor([0.2250, 0.3375, 0.2360, 0.0930, 0.3440, 0.3156, 0.1937, 0.3585, 0.1761])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = DDIMScheduler(tensor_format="pt")

        ddim = DDIM(unet=unet, noise_scheduler=noise_scheduler)
        image = ddim(generator=generator, eta=0.0)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.7383, -0.7385, -0.7298, -0.7364, -0.7414, -0.7239, -0.6737, -0.6813, -0.7068]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_pndm_cifar10(self):
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = PNDMScheduler(tensor_format="pt")

        pndm = PNDM(unet=unet, noise_scheduler=noise_scheduler)
        image = pndm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.7888, -0.7870, -0.7759, -0.7823, -0.8014, -0.7608, -0.6818, -0.7130, -0.7471]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_ldm_text2img(self):
        model_id = "fusing/latent-diffusion-text2im-large"
        ldm = LatentDiffusion.from_pretrained(model_id)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, num_inference_steps=20)

        image_slice = image[0, -1, -3:, -3:].cpu()
        print(image_slice.shape)

        assert image.shape == (1, 3, 256, 256)
        expected_slice = torch.tensor([0.7295, 0.7358, 0.7256, 0.7435, 0.7095, 0.6884, 0.7325, 0.6921, 0.6458])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_glide_text2img(self):
        model_id = "fusing/glide-base"
        glide = GLIDE.from_pretrained(model_id)

        prompt = "a pencil sketch of a corgi"
        generator = torch.manual_seed(0)
        image = glide(prompt, generator=generator, num_inference_steps_upscale=20)

        image_slice = image[0, :3, :3, -1].cpu()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = torch.tensor([0.7119, 0.7073, 0.6460, 0.7780, 0.7423, 0.6926, 0.7378, 0.7189, 0.7784])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    def test_module_from_pipeline(self):
        model = DiffWave(num_res_layers=4)
        noise_scheduler = DDPMScheduler(timesteps=12)

        bddm = BDDM(model, noise_scheduler)

        # check if the library name for the diffwave moduel is set to pipeline module
        self.assertTrue(bddm.config["diffwave"][0] == "pipeline_bddm")

        # check if we can save and load the pipeline
        with tempfile.TemporaryDirectory() as tmpdirname:
            bddm.save_pretrained(tmpdirname)
            _ = BDDM.from_pretrained(tmpdirname)
            # check if the same works using the DifusionPipeline class
            _ = DiffusionPipeline.from_pretrained(tmpdirname)
