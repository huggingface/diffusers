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

import copy
import re
import unittest

import numpy as np
import torch

from diffusers import ControlNetXSAddon, UNet2DConditionModel, UNetControlNetXSModel
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


logger = logging.get_logger(__name__)

enable_full_determinism()


class UNetControlNetXSModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNetControlNetXSModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32)).to(torch_device)
        controlnet_cond = floats_tensor((batch_size, 3, 256, 256)).to(torch_device)
        conditioning_scale = 1

        return {
            "sample": noise,
            "timestep": time_step,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": conditioning_scale,
        }

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
            "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
            "block_out_channels": (4, 8),
            "norm_num_groups": 1,
            "cross_attention_dim": 32,
            "transformer_layers_per_block": 1,
            "num_attention_heads": 8,
            "upcast_attention": False,
            "ctrl_time_embedding_input_dim": 4,
            "ctrl_block_out_channels": [4, 8],
            "ctrl_attention_head_dim": 8,
            "ctrl_max_norm_num_groups": 1,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def get_dummy_unet(self):
        """For some tests we also need the underlying UNet. For these, we'll build the UNetControlNetXSModel from the UNet"""
        return UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
            use_linear_projection=True,
        )

    def test_from_unet2d(self):
        unet = self.get_dummy_unet()
        controlnet = ControlNetXSAddon.from_unet(unet, size_ratio=1)

        model = UNetControlNetXSModel.from_unet2d(unet, controlnet)
        model_state_dict = model.state_dict()

        def is_decomposed(module_name):
            return "down_block" in module_name or "up_block" in module_name

        def block_to_subblock_name(param_name):
            """
            Map name of a param from 'block notation' as in UNet to 'subblock notation' as in UNetControlNetXS
            e.g. 'down_blocks.1.attentions.0.proj_in.weight' ->  'base_down_subblocks.3.attention.proj_in.weight'
            """
            param_name = param_name.replace("down_blocks", "base_down_subblocks")
            param_name = param_name.replace("up_blocks", "base_up_subblocks")

            numbers = re.findall(r"\d+", param_name)
            block_idx, module_idx = int(numbers[0]), int(numbers[1])

            layers_per_block = 2
            subblocks_per_block = layers_per_block + 1  # include down/upsampler

            if "downsampler" in param_name or "upsampler" in param_name:
                subblock_idx = block_idx * subblocks_per_block + layers_per_block
            else:
                subblock_idx = block_idx * subblocks_per_block + module_idx

            param_name = re.sub(r"\d", str(subblock_idx), param_name, count=1)
            param_name = re.sub(r"resnets\.\d+", "resnet", param_name)  # eg resnets.1 -> resnet
            param_name = re.sub(r"attentions\.\d+", "attention", param_name)  # eg attentions.1 -> attention
            param_name = re.sub(r"downsamplers\.\d+", "downsampler", param_name)  # eg attentions.1 -> attention
            param_name = re.sub(r"upsamplers\.\d+", "upsampler", param_name)  # eg attentions.1 -> attention

            return param_name

        for param_name, param_value in unet.named_parameters():
            if is_decomposed(param_name):
                # check unet modules that were decomposed
                self.assertTrue(torch.equal(model_state_dict[block_to_subblock_name(param_name)], param_value))
            else:
                # check unet modules that were copied as is
                self.assertTrue(torch.equal(model_state_dict["base_" + param_name], param_value))

        # check controlnet
        for param_name, param_value in controlnet.named_parameters():
            self.assertTrue(torch.equal(model_state_dict["control_addon." + param_name], param_value))

    def test_freeze_unet2d(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = UNetControlNetXSModel(**init_dict)
        model.freeze_unet2d_params()

        for param_name, param_value in model.named_parameters():
            if "control_addon" not in param_name:
                self.assertFalse(param_value.requires_grad)
            else:
                self.assertTrue(param_value.requires_grad)

    def test_gradient_checkpointing_is_applied(self):
        model_class_copy = copy.copy(UNetControlNetXSModel)

        modules_with_gc_enabled = {}

        # now monkey patch the following function:
        #     def _set_gradient_checkpointing(self, module, value=False):
        #         if hasattr(module, "gradient_checkpointing"):
        #             module.gradient_checkpointing = value

        def _set_gradient_checkpointing_new(self, module, value=False):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                modules_with_gc_enabled[module.__class__.__name__] = True

        model_class_copy._set_gradient_checkpointing = _set_gradient_checkpointing_new

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = model_class_copy(**init_dict)

        model.enable_gradient_checkpointing()

        EXPECTED_SET = {
            "Transformer2DModel",
            "UNetMidBlock2DCrossAttn",
            "CrossAttnDownSubBlock2D",
            "DownSubBlock2D",
            "CrossAttnUpSubBlock2D",
        }

        assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    def test_forward_no_control(self):
        unet = self.get_dummy_unet()
        controlnet = ControlNetXSAddon.from_unet(unet, size_ratio=1)

        model = UNetControlNetXSModel.from_unet2d(unet, controlnet)

        unet = unet.to(torch_device)
        model = model.to(torch_device)

        input_ = self.dummy_input

        control_specific_input = ["controlnet_cond", "conditioning_scale"]
        input_for_unet = {k: v for k, v in input_.items() if k not in control_specific_input}

        with torch.no_grad():
            unet_output = unet(**input_for_unet).sample.cpu()
            unet_controlnet_output = model(**input_, do_control=False).sample.cpu()

        assert np.abs(unet_output.flatten() - unet_controlnet_output.flatten()).max() < 1e-5

    def test_time_embedding_mixing(self):
        unet = self.get_dummy_unet()
        controlnet = ControlNetXSAddon.from_unet(unet, size_ratio=1)
        controlnet_mix_time = ControlNetXSAddon.from_unet(unet, size_ratio=1, time_embedding_mix=0.5)

        model = UNetControlNetXSModel.from_unet2d(unet, controlnet)
        model_mix_time = UNetControlNetXSModel.from_unet2d(unet, controlnet_mix_time)

        unet = unet.to(torch_device)
        model = model.to(torch_device)
        model_mix_time = model_mix_time.to(torch_device)

        input_ = self.dummy_input

        with torch.no_grad():
            output = model(**input_).sample
            output_mix_time = model_mix_time(**input_).sample

        assert output.shape == output_mix_time.shape
