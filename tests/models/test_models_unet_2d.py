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

import gc
import math
import tracemalloc
import unittest

import torch

from diffusers import UNet2DConditionModel, UNet2DModel
from diffusers.utils import (
    floats_tensor,
    load_hf_numpy,
    logging,
    require_torch_gpu,
    slow,
    torch_all_close,
    torch_device,
)
from parameterized import parameterized

from ..test_modeling_common import ModelTesterMixin


logger = logging.get_logger(__name__)
torch.backends.cuda.matmul.allow_tf32 = False


class Unet2DModelTests(ModelTesterMixin, unittest.TestCase):
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
        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(torch_device != "cuda", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate(self):
        model, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model.to(torch_device)
        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(torch_device != "cuda", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate_wont_change_results(self):
        # by defautl model loading will use accelerate as `low_cpu_mem_usage=True`
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_accelerate.to(torch_device)
        model_accelerate.eval()

        noise = torch.randn(
            1,
            model_accelerate.config.in_channels,
            model_accelerate.config.sample_size,
            model_accelerate.config.sample_size,
            generator=torch.manual_seed(0),
        )
        noise = noise.to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)

        arr_accelerate = model_accelerate(noise, time_step)["sample"]

        # two models don't need to stay in the device at the same time
        del model_accelerate
        torch.cuda.empty_cache()
        gc.collect()

        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, low_cpu_mem_usage=False
        )
        model_normal_load.to(torch_device)
        model_normal_load.eval()
        arr_normal_load = model_normal_load(noise, time_step)["sample"]

        assert torch_all_close(arr_accelerate, arr_normal_load, rtol=1e-3)

    @unittest.skipIf(torch_device != "cuda", "This test is supposed to run on GPU")
    def test_memory_footprint_gets_reduced(self):
        torch.cuda.empty_cache()
        gc.collect()

        tracemalloc.start()
        # by defautl model loading will use accelerate as `low_cpu_mem_usage=True`
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_accelerate.to(torch_device)
        model_accelerate.eval()
        _, peak_accelerate = tracemalloc.get_traced_memory()

        del model_accelerate
        torch.cuda.empty_cache()
        gc.collect()

        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, low_cpu_mem_usage=False
        )
        model_normal_load.to(torch_device)
        model_normal_load.eval()
        _, peak_normal = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        assert peak_accelerate < peak_normal

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()
        model.to(torch_device)

        noise = torch.randn(
            1,
            model.config.in_channels,
            model.config.sample_size,
            model.config.sample_size,
            generator=torch.manual_seed(0),
        )
        noise = noise.to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-13.3258, -20.1100, -15.9873, -17.6617, -23.0596, -17.9419, -13.3675, -16.1889, -12.3800])
        # fmt: on

        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-3))


class UNet2DConditionModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DConditionModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skipIf(torch_device == "mps", "Gradient checkpointing skipped on MPS")
    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()

        labels = torch.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        for name, param in named_params.items():
            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=5e-5))


class NCSNppModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor(batch_size * [10]).to(dtype=torch.int32, device=torch_device)

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

    @slow
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

    @slow
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
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-4836.2231, -6487.1387, -3816.7969, -7964.9253, -10966.2842, -20043.6016, 8137.0571, 2340.3499, 544.6114])
        # fmt: on

        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-2))

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
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on

        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-2))

    def test_forward_with_norm_groups(self):
        # not required for this model
        pass


@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch_dtype, revision=revision
        )
        model.to(torch_device).eval()

        return model

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        hidden_states = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return hidden_states

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4424, 0.1510, -0.1937, 0.2118, 0.3746, -0.3957, 0.0160, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.2540, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
            # fmt: on
        ]
    )
    def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_compvis_sd_v1_4_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4430, 0.1570, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.3430, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.1750]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
            # fmt: on
        ]
    )
    def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187, -0.1676, -0.1395, -0.5972]],
            [17, 0.55, [-0.1290, -0.2588, 0.0551, -0.0916, 0.3286, 0.0238, -0.3669, 0.0322]],
            [8, 0.89, [-0.5283, 0.1198, 0.0870, -0.1141, 0.9189, -0.0150, 0.5474, 0.4319]],
            [3, 1000, [-0.5601, 0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.0280, -1.0020]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_compvis_sd_v1_5_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.4120, -0.5621, -0.0397, -1.0430]],
            [9, 1000, [0.1600, 0.7303, -1.0556, -0.3515, -0.7440, -1.2037, -1.8149, -1.8931]],
            # fmt: on
        ]
    )
    def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting")
        latents = self.get_latents(seed, shape=(4, 9, 64, 64))
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, -0.4172, -0.1388, 1.1387]],
            [17, 0.55, [0.0975, -0.2856, -0.3508, -0.4600, 0.3376, 0.2930, -0.2747, -0.7026]],
            [8, 0.89, [-0.0952, 0.0183, -0.5825, -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]],
            [3, 1000, [0.4790, 0.4949, -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_compvis_sd_inpaint_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting", fp16=True)
        latents = self.get_latents(seed, shape=(4, 9, 64, 64), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)
