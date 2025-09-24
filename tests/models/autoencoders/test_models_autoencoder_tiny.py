# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import copy
import gc
import unittest

import torch
from parameterized import parameterized

from diffusers import AutoencoderTiny

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_hf_numpy,
    slow,
    torch_all_close,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderTinyTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AutoencoderTiny
    main_input_name = "sample"
    base_precision = 1e-2

    def get_autoencoder_tiny_config(self, block_out_channels=None):
        block_out_channels = (len(block_out_channels) * [32]) if block_out_channels is not None else [32, 32]
        init_dict = {
            "in_channels": 3,
            "out_channels": 3,
            "encoder_block_out_channels": block_out_channels,
            "decoder_block_out_channels": block_out_channels,
            "num_encoder_blocks": [b // min(block_out_channels) for b in block_out_channels],
            "num_decoder_blocks": [b // min(block_out_channels) for b in reversed(block_out_channels)],
        }
        return init_dict

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
        init_dict = self.get_autoencoder_tiny_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skip("Model doesn't yet support smaller resolution.")
    def test_enable_disable_tiling(self):
        pass

    @unittest.skip("Test not supported.")
    def test_outputs_equivalence(self):
        pass

    @unittest.skip("Test not supported.")
    def test_forward_with_norm_groups(self):
        pass

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"DecoderTiny", "EncoderTiny"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_effective_gradient_checkpointing(self):
        if not self.model_class._supports_gradient_checkpointing:
            return  # Skip test if model does not support gradient checkpointing

        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        inputs_dict_copy = copy.deepcopy(inputs_dict)
        torch.manual_seed(0)
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
        torch.manual_seed(0)
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict_copy).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-3)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())

        for name, param in named_params.items():
            if "encoder.layers" in name:
                continue
            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=3e-2))

    @unittest.skip(
        "The forward pass of AutoencoderTiny creates a torch.float32 tensor. This causes inference in compute_dtype=torch.bfloat16 to fail. To fix:\n"
        "1. Change the forward pass to be dtype agnostic.\n"
        "2. Unskip this test."
    )
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip(
        "The forward pass of AutoencoderTiny creates a torch.float32 tensor. This causes inference in compute_dtype=torch.bfloat16 to fail. To fix:\n"
        "1. Change the forward pass to be dtype agnostic.\n"
        "2. Unskip this test."
    )
    def test_layerwise_casting_memory(self):
        pass


@slow
class AutoencoderTinyIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="hf-internal-testing/taesd-diffusers", fp16=False):
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch_dtype)
        model.to(torch_device).eval()
        return model

    @parameterized.expand(
        [
            [(1, 4, 73, 97), (1, 3, 584, 776)],
            [(1, 4, 97, 73), (1, 3, 776, 584)],
            [(1, 4, 49, 65), (1, 3, 392, 520)],
            [(1, 4, 65, 49), (1, 3, 520, 392)],
            [(1, 4, 49, 49), (1, 3, 392, 392)],
        ]
    )
    def test_tae_tiling(self, in_shape, out_shape):
        model = self.get_sd_vae_model()
        model.enable_tiling()
        with torch.no_grad():
            zeros = torch.zeros(in_shape).to(torch_device)
            dec = model.decode(zeros).sample
            assert dec.shape == out_shape

    def test_stable_diffusion(self):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed=33)

        with torch.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor([0.0093, 0.6385, -0.1274, 0.1631, -0.1762, 0.5232, -0.3108, -0.0382])

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand([(True,), (False,)])
    def test_tae_roundtrip(self, enable_tiling):
        # load the autoencoder
        model = self.get_sd_vae_model()
        if enable_tiling:
            model.enable_tiling()

        # make a black image with a white square in the middle,
        # which is large enough to split across multiple tiles
        image = -torch.ones(1, 3, 1024, 1024, device=torch_device)
        image[..., 256:768, 256:768] = 1.0

        # round-trip the image through the autoencoder
        with torch.no_grad():
            sample = model(image).sample

        # the autoencoder reconstruction should match original image, sorta
        def downscale(x):
            return torch.nn.functional.avg_pool2d(x, model.spatial_scale_factor)

        assert torch_all_close(downscale(sample), downscale(image), atol=0.125)
