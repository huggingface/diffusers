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
import os
import tracemalloc
import unittest

import torch

from diffusers import UNet2DConditionModel, UNet2DModel
from diffusers.utils import floats_tensor, slow, torch_device

from .test_modeling_common import ModelTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


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
#            output = model(noise, time_step).sample
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
        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(torch_device == "cpu", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate(self):
        model, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, device_map="auto"
        )
        model.to(torch_device)
        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(torch_device == "cpu", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate_wont_change_results(self):
        model_accelerate, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, device_map="auto"
        )
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

        model_normal_load, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_normal_load.to(torch_device)
        model_normal_load.eval()
        arr_normal_load = model_normal_load(noise, time_step)["sample"]

        assert torch.allclose(arr_accelerate, arr_normal_load, rtol=1e-3)

    @unittest.skipIf(torch_device == "cpu", "This test is supposed to run on GPU")
    def test_memory_footprint_gets_reduced(self):
        torch.cuda.empty_cache()
        gc.collect()

        tracemalloc.start()
        model_accelerate, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, device_map="auto"
        )
        model_accelerate.to(torch_device)
        model_accelerate.eval()
        _, peak_accelerate = tracemalloc.get_traced_memory()

        del model_accelerate
        torch.cuda.empty_cache()
        gc.collect()

        model_normal_load, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
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

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-3))


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

    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        torch.use_deterministic_algorithms(True)

        # from torch docs: "A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater,
        # unless the environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8 is set."
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()
        out.sum().backward()

        # now we save the output and parameter gradients that we will use for comparison purposes with
        # the non-checkpointed run.
        output_not_checkpointed = out.data.clone()
        grad_not_checkpointed = {}
        for name, param in model.named_parameters():
            grad_not_checkpointed[name] = param.grad.data.clone()

        model.enable_gradient_checkpointing()
        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()
        out.sum().backward()

        # now we save the output and parameter gradients that we will use for comparison purposes with
        # the non-checkpointed run.
        output_checkpointed = out.data.clone()
        grad_checkpointed = {}
        for name, param in model.named_parameters():
            grad_checkpointed[name] = param.grad.data.clone()

        # compare the output and parameters gradients
        self.assertTrue((output_checkpointed == output_not_checkpointed).all())
        for name in grad_checkpointed:
            self.assertTrue(torch.allclose(grad_checkpointed[name], grad_not_checkpointed[name], atol=5e-5))

        # disable deterministic behavior for gradient checkpointing
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.use_deterministic_algorithms(False)


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
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))

    def test_forward_with_norm_groups(self):
        # not required for this model
        pass
