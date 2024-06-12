# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import numpy as np
import torch

from diffusers.models import ModelMixin, UNet3DConditionModel
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, skip_mps, torch_device

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


enable_full_determinism()

logger = logging.get_logger(__name__)


@skip_mps
class UNet3DConditionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet3DConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (16, 16)

        noise = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 4, 16, 16)

    @property
    def output_shape(self):
        return (4, 4, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (4, 8),
            "norm_num_groups": 4,
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 8,
            "attention_head_dim": 2,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 16,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_enable_works(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        model.enable_xformers_memory_efficient_attention()

        assert (
            model.mid_block.attentions[0].transformer_blocks[0].attn1.processor.__class__.__name__
            == "XFormersAttnProcessor"
        ), "xformers is not enabled"

    # Overriding to set `norm_num_groups` needs to be different for this model.
    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["block_out_channels"] = (32, 64)
        init_dict["norm_num_groups"] = 32

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    # Overriding since the UNet3D outputs a different structure.
    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            # Warmup pass when using mps (see #372)
            if torch_device == "mps" and isinstance(model, ModelMixin):
                model(**self.dummy_input)

            first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first.sample

            second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second.sample

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    def test_model_attention_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = 8

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model.set_attention_slice("auto")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice("max")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice(2)
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

    def test_feed_forward_chunking(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["block_out_channels"] = (32, 64)
        init_dict["norm_num_groups"] = 32

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)[0]

        model.enable_forward_chunking()
        with torch.no_grad():
            output_2 = model(**inputs_dict)[0]

        self.assertEqual(output.shape, output_2.shape, "Shape doesn't match")
        assert np.abs(output.cpu() - output_2.cpu()).max() < 1e-2
