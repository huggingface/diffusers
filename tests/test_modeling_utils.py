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
import math
import tempfile
import unittest

import numpy as np
import torch

import PIL
from diffusers import UNet2DConditionModel  # noqa: F401 TODO(Patrick) - need to write tests with it
from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    KarrasVePipeline,
    KarrasVeScheduler,
    LDMPipeline,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMPipeline,
    PNDMScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    StableDiffusionPipeline,
    UNet2DModel,
    VQModel,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.testing_utils import floats_tensor, slow, torch_device
from diffusers.training_utils import EMAModel


torch.backends.cuda.matmul.allow_tf32 = False


class SampleObject(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
    ):
        pass


class ConfigTester(unittest.TestCase):
    def test_load_not_from_mixin(self):
        with self.assertRaises(ValueError):
            ConfigMixin.from_config("dummy_path")

    def test_register_to_config(self):
        obj = SampleObject()
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # init ignore private arguments
        obj = SampleObject(_name_or_path="lalala")
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # can override default
        obj = SampleObject(c=6)
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == 6
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # can use positional arguments.
        obj = SampleObject(1, c=6)
        config = obj.config
        assert config["a"] == 1
        assert config["b"] == 5
        assert config["c"] == 6
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

    def test_save_load(self):
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


class ModelTesterMixin:
    def test_from_pretrained_save_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        with torch.no_grad():
            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image["sample"]

            new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image["sample"]

        max_diff = (image - new_image).abs().sum().item()
        self.assertLessEqual(max_diff, 5e-5, "Models give different forward passes")

    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first["sample"]

            second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second["sample"]

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output["sample"]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["sample", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model_from_config(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # test if the model can be loaded from the config
        # and has all the expected shape
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_config(tmpdirname)
            new_model = self.model_class.from_config(tmpdirname)
            new_model.to(torch_device)
            new_model.eval()

        # check if all paramters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            self.assertEqual(param_1.shape, param_2.shape)

        with torch.no_grad():
            output_1 = model(**inputs_dict)

            if isinstance(output_1, dict):
                output_1 = output_1["sample"]

            output_2 = new_model(**inputs_dict)

            if isinstance(output_2, dict):
                output_2 = output_2["sample"]

        self.assertEqual(output_1.shape, output_2.shape)

    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output["sample"]

        noise = torch.randn((inputs_dict["sample"].shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()

    def test_ema_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        ema_model = EMAModel(model, device=torch_device)

        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output["sample"]

        noise = torch.randn((inputs_dict["sample"].shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model)


class UnetModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "attention_head_dim": None,
            "out_channels": 3,
            "in_channels": 3,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict


#    TODO(Patrick) - Re-add this test after having correctly added the final VE checkpoints
#    def test_output_pretrained(self):
#        model = UNet2DModel.from_pretrained("fusing/ddpm_dummy_update", subfolder="unet")
#        model.eval()
#
#        torch.manual_seed(0)
#        if torch.cuda.is_available():
#            torch.cuda.manual_seed_all(0)
#
#        noise = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
#        time_step = torch.tensor([10])
#
#        with torch.no_grad():
#            output = model(noise, time_step)["sample"]
#
#        output_slice = output[0, -1, -3:, -3:].flatten()
# fmt: off
#        expected_output_slice = torch.tensor([0.2891, -0.1899, 0.2595, -0.6214, 0.0968, -0.2622, 0.4688, 0.1311, 0.0053])
# fmt: on
#        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class UNetLDMModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": (32, 64),
            "attention_head_dim": 32,
            "down_block_types": ("DownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "UpBlock2D"),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)

        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)["sample"]

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        time_step = torch.tensor([10] * noise.shape[0])

        with torch.no_grad():
            output = model(noise, time_step)["sample"]

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-13.3258, -20.1100, -15.9873, -17.6617, -23.0596, -17.9419, -13.3675, -16.1889, -12.3800])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


#    TODO(Patrick) - Re-add this test after having cleaned up LDM
#    def test_output_pretrained_spatial_transformer(self):
#        model = UNetLDMModel.from_pretrained("fusing/unet-ldm-dummy-spatial")
#        model.eval()
#
#        torch.manual_seed(0)
#        if torch.cuda.is_available():
#            torch.cuda.manual_seed_all(0)
#
#        noise = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
#        context = torch.ones((1, 16, 64), dtype=torch.float32)
#        time_step = torch.tensor([10] * noise.shape[0])
#
#        with torch.no_grad():
#            output = model(noise, time_step, context=context)
#
#        output_slice = output[0, -1, -3:, -3:].flatten()
# fmt: off
#        expected_output_slice = torch.tensor([61.3445, 56.9005, 29.4339, 59.5497, 60.7375, 34.1719, 48.1951, 42.6569, 25.0890])
# fmt: on
#
#        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))
#


class NCSNppModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(torch_device)

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64, 64, 64],
            "in_channels": 3,
            "layers_per_block": 1,
            "out_channels": 3,
            "time_embedding_type": "fourier",
            "norm_eps": 1e-6,
            "mid_block_scale_factor": math.sqrt(2.0),
            "norm_num_groups": None,
            "down_block_types": [
                "SkipDownBlock2D",
                "AttnSkipDownBlock2D",
                "SkipDownBlock2D",
                "SkipDownBlock2D",
            ],
            "up_block_types": [
                "SkipUpBlock2D",
                "SkipUpBlock2D",
                "AttnSkipUpBlock2D",
                "SkipUpBlock2D",
            ],
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        inputs = self.dummy_input
        noise = floats_tensor((4, 3) + (256, 256)).to(torch_device)
        inputs["sample"] = noise
        image = model(**inputs)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained_ve_mid(self):
        model = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256")
        model.to(torch_device)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        batch_size = 4
        num_channels = 3
        sizes = (256, 256)

        noise = torch.ones((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [1e-4]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)["sample"]

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-4836.2231, -6487.1387, -3816.7969, -7964.9253, -10966.2842, -20043.6016, 8137.0571, 2340.3499, 544.6114])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))

    def test_output_pretrained_ve_large(self):
        model = UNet2DModel.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy-update")
        model.to(torch_device)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = torch.ones((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [1e-4]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)["sample"]

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class VQModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = VQModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 3,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = VQModel.from_pretrained("fusing/vqgan-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = VQModel.from_pretrained("fusing/vqgan-dummy")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        with torch.no_grad():
            output = model(image)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0153, -0.4044, -0.1880, -0.5161, -0.2418, -0.4072, -0.1612, -0.0633, -0.0143])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class AutoencoderKLTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        with torch.no_grad():
            output = model(image, sample_posterior=True)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-4.0078e-01, -3.8304e-04, -1.2681e-01, -1.1462e-01, 2.0095e-01, 1.0893e-01, -8.8248e-02, -3.0361e-01, -9.8646e-03])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class PipelineTesterMixin(unittest.TestCase):
    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        schedular = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, schedular)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator, output_type="numpy")["sample"]
        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator, output_type="numpy")["sample"]

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        ddpm = DDPMPipeline.from_pretrained(model_path)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path)

        ddpm.scheduler.num_timesteps = 10
        ddpm_from_hub.scheduler.num_timesteps = 10

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator, output_type="numpy")["sample"]
        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy")["sample"]

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        pipe = DDIMPipeline.from_pretrained(model_path)

        generator = torch.manual_seed(0)
        images = pipe(generator=generator, output_type="numpy")["sample"]
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(generator=generator, output_type="pil")["sample"]
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(generator=generator)["sample"]
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    @slow
    def test_ddpm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_config(model_id)
        scheduler = scheduler.set_format("pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.41995, 0.35885, 0.19385, 0.38475, 0.3382, 0.2647, 0.41545, 0.3582, 0.33845])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddim_lsun(self):
        model_id = "google/ddpm-ema-bedroom-256"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_config(model_id)

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.00605, 0.0201, 0.0344, 0.00235, 0.00185, 0.00025, 0.00215, 0.0, 0.00685])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler(tensor_format="pt")

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.17235, 0.16175, 0.16005, 0.16255, 0.1497, 0.1513, 0.15045, 0.1442, 0.1453])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_pndm_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = PNDMScheduler(tensor_format="pt")

        pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 0.14045, 0.13115, 0.12175, 0.125])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="numpy")[
            "sample"
        ]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.9256, 0.9340, 0.8933, 0.9361, 0.9113, 0.8727, 0.9122, 0.8745, 0.8099])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_text2img_fast(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, num_inference_steps=1, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.3163, 0.8670, 0.6465, 0.1865, 0.6291, 0.5139, 0.2824, 0.3723, 0.4344])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion(self):
        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1-diffusers")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        with torch.autocast("cuda"):
            output = sd_pipe(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="np"
            )

        image = output["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8887, 0.915, 0.91, 0.894, 0.909, 0.912, 0.919, 0.925, 0.883])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_stable_diffusion_fast_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1-diffusers")

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        sd_pipe.scheduler = scheduler

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device=torch_device).manual_seed(0)

        with torch.autocast("cuda"):
            output = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="numpy")
        image = output["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8354, 0.83, 0.866, 0.838, 0.8315, 0.867, 0.836, 0.8584, 0.869])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @slow
    def test_score_sde_ve_pipeline(self):
        model_id = "google/ncsnpp-church-256"
        model = UNet2DModel.from_pretrained(model_id)

        scheduler = ScoreSdeVeScheduler.from_config(model_id)

        sde_ve = ScoreSdeVePipeline(unet=model, scheduler=scheduler)

        torch.manual_seed(0)
        image = sde_ve(num_inference_steps=300, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)

        expected_slice = np.array([0.64363, 0.5868, 0.3031, 0.2284, 0.7409, 0.3216, 0.25643, 0.6557, 0.2633])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ldm_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.4399, 0.44975, 0.46825, 0.474, 0.4359, 0.4581, 0.45095, 0.4341, 0.4447])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_ddpm_ddim_equality(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler(tensor_format="pt")
        ddim_scheduler = DDIMScheduler(tensor_format="pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)

        generator = torch.manual_seed(0)
        ddpm_image = ddpm(generator=generator, output_type="numpy")["sample"]

        generator = torch.manual_seed(0)
        ddim_image = ddim(generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy")["sample"]

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_image - ddim_image).max() < 1e-1

    @unittest.skip("(Anton) The test is failing for large batch sizes, needs investigation")
    def test_ddpm_ddim_equality_batched(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler(tensor_format="pt")
        ddim_scheduler = DDIMScheduler(tensor_format="pt")

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)

        generator = torch.manual_seed(0)
        ddpm_images = ddpm(batch_size=4, generator=generator, output_type="numpy")["sample"]

        generator = torch.manual_seed(0)
        ddim_images = ddim(batch_size=4, generator=generator, num_inference_steps=1000, eta=1.0, output_type="numpy")[
            "sample"
        ]

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1

    @slow
    def test_karras_ve_pipeline(self):
        model_id = "google/ncsnpp-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = KarrasVeScheduler(tensor_format="pt")

        pipe = KarrasVePipeline(unet=model, scheduler=scheduler)

        generator = torch.manual_seed(0)
        image = pipe(num_inference_steps=20, generator=generator, output_type="numpy")["sample"]

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.26815, 0.1581, 0.2658, 0.23248, 0.1550, 0.2539, 0.1131, 0.1024, 0.0837])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    @unittest.skipIf(torch_device == "cpu", "Stable diffusion is supposed to run on GPU")
    def test_lms_stable_diffusion_pipeline(self):
        model_id = "CompVis/stable-diffusion-v1-1-diffusers"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
        scheduler = LMSDiscreteScheduler.from_config(model_id, subfolder="scheduler", use_auth_token=True)
        pipe.scheduler = scheduler

        prompt = "a photograph of an astronaut riding a horse"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = pipe([prompt], generator=generator, guidance_scale=7.5, num_inference_steps=10, output_type="numpy")[
            "sample"
        ]

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9077, 0.9254, 0.9181, 0.9227, 0.9213, 0.9367, 0.9399, 0.9406, 0.9024])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
