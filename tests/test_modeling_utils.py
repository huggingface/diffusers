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
    BDDMPipeline,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    GlidePipeline,
    GlideSuperResUNetModel,
    GlideTextToImageUNetModel,
    GradTTSPipeline,
    GradTTSScheduler,
    LatentDiffusionPipeline,
    NCSNpp,
    PNDMPipeline,
    PNDMScheduler,
    ScoreSdeVePipeline,
    ScoreSdeVeScheduler,
    ScoreSdeVpPipeline,
    ScoreSdeVpScheduler,
    UNetGradTTSModel,
    UNetLDMModel,
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
        expected_shape = inputs_dict["x"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["x", "timesteps"]
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
        noise = torch.randn((inputs_dict["x"].shape[0],) + self.get_output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()


class UnetModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNetModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"x": noise, "timesteps": time_step}

    @property
    def get_input_shape(self):
        return (3, 32, 32)

    @property
    def get_output_shape(self):
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
        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


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

        return {"x": noise, "timesteps": time_step, "low_res": low_res}

    @property
    def get_input_shape(self):
        return (3, 32, 32)

    @property
    def get_output_shape(self):
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
        expected_shape = inputs_dict["x"].shape
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

        return {"x": noise, "timesteps": time_step, "transformer_out": emb}

    @property
    def get_input_shape(self):
        return (3, 32, 32)

    @property
    def get_output_shape(self):
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
        expected_shape = inputs_dict["x"].shape
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
    model_class = UNetLDMModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)

        return {"x": noise, "timesteps": time_step}

    @property
    def get_input_shape(self):
        return (4, 32, 32)

    @property
    def get_output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "image_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 32,
            "num_res_blocks": 2,
            "attention_resolutions": (16,),
            "channel_mult": (1, 2),
            "num_heads": 2,
            "conv_resample": True,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNetLDMModel.from_pretrained("fusing/unet-ldm-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNetLDMModel.from_pretrained("fusing/unet-ldm-dummy")
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


class UNetGradTTSModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNetGradTTSModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_features = 32
        seq_len = 16

        noise = floats_tensor((batch_size, num_features, seq_len)).to(torch_device)
        condition = floats_tensor((batch_size, num_features, seq_len)).to(torch_device)
        mask = floats_tensor((batch_size, 1, seq_len)).to(torch_device)
        time_step = torch.tensor([10] * batch_size).to(torch_device)

        return {"x": noise, "timesteps": time_step, "mu": condition, "mask": mask}

    @property
    def get_input_shape(self):
        return (4, 32, 16)

    @property
    def get_output_shape(self):
        return (4, 32, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "dim": 64,
            "groups": 4,
            "dim_mults": (1, 2),
            "n_feats": 32,
            "pe_scale": 1000,
            "n_spks": 1,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNetGradTTSModel.from_pretrained("fusing/unet-grad-tts-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNetGradTTSModel.from_pretrained("fusing/unet-grad-tts-dummy")
        model.eval()

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        num_features = model.config.n_feats
        seq_len = 16
        noise = torch.randn((1, num_features, seq_len))
        condition = torch.randn((1, num_features, seq_len))
        mask = torch.randn((1, 1, seq_len))
        time_step = torch.tensor([10])

        with torch.no_grad():
            output = model(noise, time_step, condition, mask)

        output_slice = output[0, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0690, -0.0531, 0.0633, -0.0660, -0.0541, 0.0650, -0.0656, -0.0555, 0.0617])
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

        return {"x": noise, "timesteps": time_step}

    @property
    def get_input_shape(self):
        return (3, 32, 32)

    @property
    def get_output_shape(self):
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

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([3.1909e-07, -8.5393e-08, 4.8460e-07, -4.5550e-07, -1.3205e-06, -6.3475e-07, 9.7837e-07, 2.9974e-07, 1.2345e-06])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

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

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-8.3299e-07, -9.0431e-07, 4.0585e-08, 9.7563e-07, 1.0280e-06, 1.0133e-06, 1.4979e-06, -2.9716e-07, -6.1817e-07])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))

    def test_output_pretrained_vp(self):
        model = NCSNpp.from_pretrained("fusing/ddpm-cifar10-vp-dummy")
        model.eval()
        model.to(torch_device)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step)

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-3.9086e-07, -1.1001e-05, 1.8881e-06, 1.1106e-05, 1.6629e-06, 2.9820e-06, 8.4978e-06, 8.0253e-07, 1.5435e-06])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-3))


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
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        unet = UNetModel.from_pretrained(model_id)
        noise_scheduler = DDPMScheduler.from_config(model_id)
        noise_scheduler = noise_scheduler.set_format("pt")

        ddpm = DDPMPipeline(unet=unet, noise_scheduler=noise_scheduler)
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

        ddim = DDIMPipeline(unet=unet, noise_scheduler=noise_scheduler)
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

        pndm = PNDMPipeline(unet=unet, noise_scheduler=noise_scheduler)
        image = pndm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor(
            [-0.7888, -0.7870, -0.7759, -0.7823, -0.8014, -0.7608, -0.6818, -0.7130, -0.7471]
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
    def test_grad_tts(self):
        model_id = "fusing/grad-tts-libri-tts"
        grad_tts = GradTTSPipeline.from_pretrained(model_id)
        noise_scheduler = GradTTSScheduler()
        grad_tts.noise_scheduler = noise_scheduler

        text = "Hello world, I missed you so much."
        generator = torch.manual_seed(0)

        # generate mel spectograms using text
        mel_spec = grad_tts(text, generator=generator)

        assert mel_spec.shape == (1, 80, 143)
        expected_slice = torch.tensor(
            [-6.7584, -6.8347, -6.3293, -6.6437, -6.7233, -6.4684, -6.1187, -6.3172, -6.6890]
        )
        assert (mel_spec[0, :3, :3].cpu().flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_score_sde_ve_pipeline(self):
        torch.manual_seed(0)

        model = NCSNpp.from_pretrained("fusing/ffhq_ncsnpp")
        scheduler = ScoreSdeVeScheduler.from_config("fusing/ffhq_ncsnpp")

        sde_ve = ScoreSdeVePipeline(model=model, scheduler=scheduler)

        image = sde_ve(num_inference_steps=2)

        expected_image_sum = 3382810112.0
        expected_image_mean = 1075.366455078125

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

    def test_module_from_pipeline(self):
        model = DiffWave(num_res_layers=4)
        noise_scheduler = DDPMScheduler(timesteps=12)

        bddm = BDDMPipeline(model, noise_scheduler)

        # check if the library name for the diffwave moduel is set to pipeline module
        self.assertTrue(bddm.config["diffwave"][0] == "pipeline_bddm")

        # check if we can save and load the pipeline
        with tempfile.TemporaryDirectory() as tmpdirname:
            bddm.save_pretrained(tmpdirname)
            _ = BDDMPipeline.from_pretrained(tmpdirname)
            # check if the same works using the DifusionPipeline class
            _ = DiffusionPipeline.from_pretrained(tmpdirname)
