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
            new_image = new_model(**inputs_dict)

        max_diff = (image - new_image).abs().sum().item()
        self.assertLessEqual(max_diff, 5e-5, "Models give different forward passes")

    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            first = model(**inputs_dict)
            second = model(**inputs_dict)

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

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["sample", "timesteps"]
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
            output_2 = new_model(**inputs_dict)

        self.assertEqual(output_1.shape, output_2.shape)

    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict)
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
        noise = torch.randn((inputs_dict["sample"].shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model)


class UnetModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNetModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"sample": noise, "timesteps": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "ch": 32,
            "ch_mult": (1, 2),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "resolution": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNetModel.from_pretrained("fusing/ddpm_dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNetModel.from_pretrained("fusing/ddpm_dummy")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.resolution, model.config.resolution)
        time_step = torch.tensor([10])

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([0.2891, -0.1899, 0.2595, -0.6214, 0.0968, -0.2622, 0.4688, 0.1311, 0.0053])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))

        print("Original success!!!")

        model = UNetUnconditionalModel.from_pretrained("fusing/ddpm_dummy", ddpm=True)
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
        time_step = torch.tensor([10])

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([0.2891, -0.1899, 0.2595, -0.6214, 0.0968, -0.2622, 0.4688, 0.1311, 0.0053])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class GlideSuperResUNetTests(ModelTesterMixin, unittest.TestCase):
    model_class = GlideSuperResUNetModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 6
        sizes = (32, 32)
        low_res_size = (4, 4)

        noise = torch.randn((batch_size, num_channels // 2) + sizes).to(torch_device)
        low_res = torch.randn((batch_size, 3) + low_res_size).to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0], device=torch_device)

        return {"sample": noise, "timesteps": time_step, "low_res": low_res}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (6, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "attention_resolutions": (2,),
            "channel_mult": (1, 2),
            "in_channels": 6,
            "out_channels": 6,
            "model_channels": 32,
            "num_head_channels": 8,
            "num_heads_upsample": 1,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "resolution": 32,
            "use_scale_shift_norm": True,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

        output, _ = torch.split(output, 3, dim=1)

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_from_pretrained_hub(self):
        model, loading_info = GlideSuperResUNetModel.from_pretrained(
            "fusing/glide-super-res-dummy", output_loading_info=True
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = GlideSuperResUNetModel.from_pretrained("fusing/glide-super-res-dummy")

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, 3, 64, 64)
        low_res = torch.randn(1, 3, 4, 4)
        time_step = torch.tensor([42] * noise.shape[0])

        with torch.no_grad():
            output = model(noise, time_step, low_res)

        output, _ = torch.split(output, 3, dim=1)
        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-22.8782, -23.2652, -15.3966, -22.8034, -23.3159, -15.5640, -15.3970, -15.4614, - 10.4370])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


class GlideTextToImageUNetModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = GlideTextToImageUNetModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)
        transformer_dim = 32
        seq_len = 16

        noise = torch.randn((batch_size, num_channels) + sizes).to(torch_device)
        emb = torch.randn((batch_size, seq_len, transformer_dim)).to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0], device=torch_device)

        return {"sample": noise, "timesteps": time_step, "transformer_out": emb}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (6, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "attention_resolutions": (2,),
            "channel_mult": (1, 2),
            "in_channels": 3,
            "out_channels": 6,
            "model_channels": 32,
            "num_head_channels": 8,
            "num_heads_upsample": 1,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "resolution": 32,
            "use_scale_shift_norm": True,
            "transformer_dim": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

        output, _ = torch.split(output, 3, dim=1)

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_from_pretrained_hub(self):
        model, loading_info = GlideTextToImageUNetModel.from_pretrained(
            "fusing/unet-glide-text2im-dummy", output_loading_info=True
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = GlideTextToImageUNetModel.from_pretrained("fusing/unet-glide-text2im-dummy")

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn((1, model.config.in_channels, model.config.resolution, model.config.resolution)).to(
            torch_device
        )
        emb = torch.randn((1, 16, model.config.transformer_dim)).to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0], device=torch_device)

        model.to(torch_device)
        with torch.no_grad():
            output = model(noise, time_step, emb)

        output, _ = torch.split(output, 3, dim=1)
        output_slice = output[0, -1, -3:, -3:].cpu().flatten()
        # fmt: off
        expected_output_slice = torch.tensor([2.7766, -10.3558, -14.9149, -0.9376, -14.9175, -17.7679, -5.5565, -12.9521, -12.9845])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


class UNetLDMModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNetUnconditionalModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"sample": noise, "timesteps": time_step}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "image_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 2,
            "attention_resolutions": (16,),
            "block_channels": (32, 64),
            "num_head_channels": 32,
            "conv_resample": True,
            "down_blocks": ("UNetResDownBlock2D", "UNetResDownBlock2D"),
            "up_blocks": ("UNetResUpBlock2D", "UNetResUpBlock2D"),
            "ldm": True,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNetUnconditionalModel.from_pretrained(
            "fusing/unet-ldm-dummy", output_loading_info=True, ldm=True
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNetUnconditionalModel.from_pretrained("fusing/unet-ldm-dummy", ldm=True)
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
        time_step = torch.tensor([10] * noise.shape[0])

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-13.3258, -20.1100, -15.9873, -17.6617, -23.0596, -17.9419, -13.3675, -16.1889, -12.3800])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_output_pretrained_spatial_transformer(self):
        model = UNetLDMModel.from_pretrained("fusing/unet-ldm-dummy-spatial")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
        context = torch.ones((1, 16, 64), dtype=torch.float32)
        time_step = torch.tensor([10] * noise.shape[0])

        with torch.no_grad():
            output = model(noise, time_step, context=context)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([61.3445, 56.9005, 29.4339, 59.5497, 60.7375, 34.1719, 48.1951, 42.6569, 25.0890])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


class NCSNppModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = NCSNpp

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(torch_device)

        return {"sample": noise, "timesteps": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "image_size": 32,
            "ch_mult": [1, 2, 2, 2],
            "nf": 32,
            "fir": True,
            "progressive": "output_skip",
            "progressive_combine": "sum",
            "progressive_input": "input_skip",
            "scale_by_sigma": True,
            "skip_rescale": True,
            "embedding_type": "fourier",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = NCSNpp.from_pretrained("fusing/cifar10-ncsnpp-ve", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained_ve_small(self):
        model = NCSNpp.from_pretrained("fusing/ncsnpp-cifar10-ve-dummy")
        model.eval()
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
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([0.1315, 0.0741, 0.0393, 0.0455, 0.0556, 0.0180, -0.0832, -0.0644, -0.0856])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))

    def test_output_pretrained_ve_large(self):
        model = NCSNpp.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy")
        model.eval()
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
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))

    def test_output_pretrained_vp(self):
        model = NCSNpp.from_pretrained("fusing/cifar10-ddpmpp-vp")
        model.eval()
        model.to(torch_device)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = torch.randn((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [9.0]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([0.3303, -0.2275, -2.8872, -0.1309, -1.2861, 3.4567, -1.0083, 2.5325, -1.3866])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class VQModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = VQModel

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
            "ch": 64,
            "out_ch": 3,
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "in_channels": 3,
            "resolution": 32,
            "z_channels": 3,
            "n_embed": 256,
            "embed_dim": 3,
            "sane_index_shape": False,
            "ch_mult": (1,),
            "dropout": 0.0,
            "double_z": False,
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

        image = torch.randn(1, model.config.in_channels, model.config.resolution, model.config.resolution)
        with torch.no_grad():
            output = model(image)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-1.1321, 0.1056, 0.3505, -0.6461, -0.2014, 0.0419, -0.5763, -0.8462, -0.4218])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class AutoEncoderKLTests(ModelTesterMixin, unittest.TestCase):
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
            "ch": 64,
            "ch_mult": (1,),
            "embed_dim": 4,
            "in_channels": 3,
            "num_res_blocks": 1,
            "out_ch": 3,
            "resolution": 32,
            "z_channels": 4,
            "attn_resolutions": [],
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

        image = torch.randn(1, model.config.in_channels, model.config.resolution, model.config.resolution)
        with torch.no_grad():
            output = model(image, sample_posterior=True)

        output_slice = output[0, -1, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0814, -0.0229, -0.1320, -0.4123, -0.0366, -0.3473, 0.0438, -0.1662, 0.1750])
        # fmt: on
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


class PipelineTesterMixin(unittest.TestCase):
    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
        schedular = DDPMScheduler(timesteps=10)

        ddpm = DDPMPipeline(model, schedular)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "fusing/ddpm-cifar10"

        ddpm = DDPMPipeline.from_pretrained(model_path)
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
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = DDPMScheduler.from_config(model_id)
        noise_scheduler = noise_scheduler.set_format("pt")

        ddpm = DDPMPipeline(unet=unet, noise_scheduler=noise_scheduler)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.5712, -0.6215, -0.5953, -0.5438, -0.4775, -0.4539, -0.5172, -0.4872, -0.5105]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = DDIMScheduler(tensor_format="pt")

        ddim = DDIMPipeline(unet=unet, noise_scheduler=noise_scheduler)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.6553, -0.6765, -0.6799, -0.6749, -0.7006, -0.6974, -0.6991, -0.7116, -0.7094]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_pndm_cifar10(self):
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = PNDMScheduler(tensor_format="pt")

        pndm = PNDMPipeline(unet=unet, noise_scheduler=noise_scheduler)
        generator = torch.manual_seed(0)
        image = pndm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.6872, -0.7071, -0.7188, -0.7057, -0.7515, -0.7191, -0.7377, -0.7565, -0.7500]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    @unittest.skip("Skipping for now as it takes too long")
    def test_ldm_text2img(self):
        model_id = "fusing/latent-diffusion-text2im-large"
        ldm = LatentDiffusionPipeline.from_pretrained(model_id)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, num_inference_steps=20)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 256, 256)
        expected_slice = torch.tensor([0.7295, 0.7358, 0.7256, 0.7435, 0.7095, 0.6884, 0.7325, 0.6921, 0.6458])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_ldm_text2img_fast(self):
        model_id = "fusing/latent-diffusion-text2im-large"
        ldm = LatentDiffusionPipeline.from_pretrained(model_id)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        image = ldm([prompt], generator=generator, num_inference_steps=1)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 256, 256)
        expected_slice = torch.tensor([0.3163, 0.8670, 0.6465, 0.1865, 0.6291, 0.5139, 0.2824, 0.3723, 0.4344])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_glide_text2img(self):
        model_id = "fusing/glide-base"
        glide = GlidePipeline.from_pretrained(model_id)

        prompt = "a pencil sketch of a corgi"
        generator = torch.manual_seed(0)
        image = glide(prompt, generator=generator, num_inference_steps_upscale=20)

        image_slice = image[0, :3, :3, -1].cpu()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = torch.tensor([0.7119, 0.7073, 0.6460, 0.7780, 0.7423, 0.6926, 0.7378, 0.7189, 0.7784])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_score_sde_ve_pipeline(self):
        model = NCSNpp.from_pretrained("fusing/ffhq_ncsnpp")
        scheduler = ScoreSdeVeScheduler.from_config("fusing/ffhq_ncsnpp")

        sde_ve = ScoreSdeVePipeline(model=model, scheduler=scheduler)

        torch.manual_seed(0)
        image = sde_ve(num_inference_steps=2)

        expected_image_sum = 3382849024.0
        expected_image_mean = 1075.3788

        assert (image.abs().sum() - expected_image_sum).abs().cpu().item() < 1e-2
        assert (image.abs().mean() - expected_image_mean).abs().cpu().item() < 1e-4

    @slow
    def test_score_sde_vp_pipeline(self):
        model = NCSNpp.from_pretrained("fusing/cifar10-ddpmpp-vp")
        scheduler = ScoreSdeVpScheduler.from_config("fusing/cifar10-ddpmpp-vp")

        sde_vp = ScoreSdeVpPipeline(model=model, scheduler=scheduler)

        torch.manual_seed(0)
        image = sde_vp(num_inference_steps=10)

        expected_image_sum = 4183.2012
        expected_image_mean = 1.3617

        assert (image.abs().sum() - expected_image_sum).abs().cpu().item() < 1e-2
        assert (image.abs().mean() - expected_image_mean).abs().cpu().item() < 1e-4

    @slow
    def test_ldm_uncond(self):
        ldm = LatentDiffusionUncondPipeline.from_pretrained("fusing/latent-diffusion-celeba-256")

        generator = torch.manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 256, 256)
        expected_slice = torch.tensor(
            [-0.1202, -0.1005, -0.0635, -0.0520, -0.1282, -0.0838, -0.0981, -0.1318, -0.1106]
        )
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2
