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

import unittest

import numpy as np
import pytest
import torch

from diffusers import UNet3DConditionModel
from diffusers.utils.import_utils import is_xformers_available

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    skip_mps,
    torch_device,
)
from ..test_modeling_common import UNetTesterMixin
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


@skip_mps
class UNet3DConditionTesterConfig(BaseModelTesterConfig):
    """Base configuration for UNet3DConditionModel testing."""

    @property
    def model_class(self):
        return UNet3DConditionModel

    @property
    def output_shape(self):
        return (4, 4, 16, 16)

    @property
    def main_input_name(self):
        return "sample"

    def get_init_dict(self):
        return {
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

    def get_dummy_inputs(self):
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (16, 16)

        return {
            "sample": floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device),
            "timestep": torch.tensor([10]).to(torch_device),
            "encoder_hidden_states": floats_tensor((batch_size, 4, 8)).to(torch_device),
        }


class TestUNet3DCondition(UNet3DConditionTesterConfig, ModelTesterMixin, UNetTesterMixin):
    # Overriding to set `norm_num_groups` needs to be different for this model.
    def test_forward_with_norm_groups(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        init_dict["block_out_channels"] = (32, 64)
        init_dict["norm_num_groups"] = 32

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        assert output is not None
        expected_shape = inputs_dict["sample"].shape
        assert output.shape == expected_shape, "Input and output shapes do not match"

    # Overriding since the UNet3D outputs a different structure.
    @torch.no_grad()
    def test_determinism(self):
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()

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
        assert max_diff <= 1e-5

    def test_feed_forward_chunking(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
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

        assert output.shape == output_2.shape, "Shape doesn't match"
        assert np.abs(output.cpu() - output_2.cpu()).max() < 1e-2


class TestUNet3DConditionAttention(UNet3DConditionTesterConfig, AttentionTesterMixin):
    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_enable_works(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)

        model.enable_xformers_memory_efficient_attention()

        assert (
            model.mid_block.attentions[0].transformer_blocks[0].attn1.processor.__class__.__name__
            == "XFormersAttnProcessor"
        ), "xformers is not enabled"

    def test_model_attention_slicing(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

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


class TestUNet3DConditionMemory(UNet3DConditionTesterConfig, MemoryTesterMixin):
    pass


class TestUNet3DConditionTraining(UNet3DConditionTesterConfig, TrainingTesterMixin):
    pass


class TestUNet3DConditionLoRA(UNet3DConditionTesterConfig, LoraTesterMixin):
    pass
