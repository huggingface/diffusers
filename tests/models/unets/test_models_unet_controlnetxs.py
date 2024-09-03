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
import unittest

import numpy as np
import torch
from torch import nn

from diffusers import ControlNetXSAdapter, UNet2DConditionModel, UNetControlNetXSModel
from diffusers.utils import logging
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, is_flaky, torch_device

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
        sizes = (16, 16)
        conditioning_image_size = (3, 32, 32)  # size of additional, unprocessed image for control-conditioning

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)
        controlnet_cond = floats_tensor((batch_size, *conditioning_image_size)).to(torch_device)
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
        return (4, 16, 16)

    @property
    def output_shape(self):
        return (4, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 16,
            "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
            "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
            "block_out_channels": (4, 8),
            "cross_attention_dim": 8,
            "transformer_layers_per_block": 1,
            "num_attention_heads": 2,
            "norm_num_groups": 4,
            "upcast_attention": False,
            "ctrl_block_out_channels": [2, 4],
            "ctrl_num_attention_heads": 4,
            "ctrl_max_norm_num_groups": 2,
            "ctrl_conditioning_embedding_out_channels": (2, 2),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def get_dummy_unet(self):
        """For some tests we also need the underlying UNet. For these, we'll build the UNetControlNetXSModel from the UNet and ControlNetXS-Adapter"""
        return UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=16,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=8,
            norm_num_groups=4,
            use_linear_projection=True,
        )

    def get_dummy_controlnet_from_unet(self, unet, **kwargs):
        """For some tests we also need the underlying ControlNetXS-Adapter. For these, we'll build the UNetControlNetXSModel from the UNet and ControlNetXS-Adapter"""
        # size_ratio and conditioning_embedding_out_channels chosen to keep model small
        return ControlNetXSAdapter.from_unet(unet, size_ratio=1, conditioning_embedding_out_channels=(2, 2), **kwargs)

    def test_from_unet(self):
        unet = self.get_dummy_unet()
        controlnet = self.get_dummy_controlnet_from_unet(unet)

        model = UNetControlNetXSModel.from_unet(unet, controlnet)
        model_state_dict = model.state_dict()

        def assert_equal_weights(module, weight_dict_prefix):
            for param_name, param_value in module.named_parameters():
                assert torch.equal(model_state_dict[weight_dict_prefix + "." + param_name], param_value)

        # # check unet
        # everything expect down,mid,up blocks
        modules_from_unet = [
            "time_embedding",
            "conv_in",
            "conv_norm_out",
            "conv_out",
        ]
        for p in modules_from_unet:
            assert_equal_weights(getattr(unet, p), "base_" + p)
        optional_modules_from_unet = [
            "class_embedding",
            "add_time_proj",
            "add_embedding",
        ]
        for p in optional_modules_from_unet:
            if hasattr(unet, p) and getattr(unet, p) is not None:
                assert_equal_weights(getattr(unet, p), "base_" + p)
        # down blocks
        assert len(unet.down_blocks) == len(model.down_blocks)
        for i, d in enumerate(unet.down_blocks):
            assert_equal_weights(d.resnets, f"down_blocks.{i}.base_resnets")
            if hasattr(d, "attentions"):
                assert_equal_weights(d.attentions, f"down_blocks.{i}.base_attentions")
            if hasattr(d, "downsamplers") and getattr(d, "downsamplers") is not None:
                assert_equal_weights(d.downsamplers[0], f"down_blocks.{i}.base_downsamplers")
        # mid block
        assert_equal_weights(unet.mid_block, "mid_block.base_midblock")
        # up blocks
        assert len(unet.up_blocks) == len(model.up_blocks)
        for i, u in enumerate(unet.up_blocks):
            assert_equal_weights(u.resnets, f"up_blocks.{i}.resnets")
            if hasattr(u, "attentions"):
                assert_equal_weights(u.attentions, f"up_blocks.{i}.attentions")
            if hasattr(u, "upsamplers") and getattr(u, "upsamplers") is not None:
                assert_equal_weights(u.upsamplers[0], f"up_blocks.{i}.upsamplers")

        # # check controlnet
        # everything expect down,mid,up blocks
        modules_from_controlnet = {
            "controlnet_cond_embedding": "controlnet_cond_embedding",
            "conv_in": "ctrl_conv_in",
            "control_to_base_for_conv_in": "control_to_base_for_conv_in",
        }
        optional_modules_from_controlnet = {"time_embedding": "ctrl_time_embedding"}
        for name_in_controlnet, name_in_unetcnxs in modules_from_controlnet.items():
            assert_equal_weights(getattr(controlnet, name_in_controlnet), name_in_unetcnxs)

        for name_in_controlnet, name_in_unetcnxs in optional_modules_from_controlnet.items():
            if hasattr(controlnet, name_in_controlnet) and getattr(controlnet, name_in_controlnet) is not None:
                assert_equal_weights(getattr(controlnet, name_in_controlnet), name_in_unetcnxs)
        # down blocks
        assert len(controlnet.down_blocks) == len(model.down_blocks)
        for i, d in enumerate(controlnet.down_blocks):
            assert_equal_weights(d.resnets, f"down_blocks.{i}.ctrl_resnets")
            assert_equal_weights(d.base_to_ctrl, f"down_blocks.{i}.base_to_ctrl")
            assert_equal_weights(d.ctrl_to_base, f"down_blocks.{i}.ctrl_to_base")
            if d.attentions is not None:
                assert_equal_weights(d.attentions, f"down_blocks.{i}.ctrl_attentions")
            if d.downsamplers is not None:
                assert_equal_weights(d.downsamplers, f"down_blocks.{i}.ctrl_downsamplers")
        # mid block
        assert_equal_weights(controlnet.mid_block.base_to_ctrl, "mid_block.base_to_ctrl")
        assert_equal_weights(controlnet.mid_block.midblock, "mid_block.ctrl_midblock")
        assert_equal_weights(controlnet.mid_block.ctrl_to_base, "mid_block.ctrl_to_base")
        # up blocks
        assert len(controlnet.up_connections) == len(model.up_blocks)
        for i, u in enumerate(controlnet.up_connections):
            assert_equal_weights(u.ctrl_to_base, f"up_blocks.{i}.ctrl_to_base")

    def test_freeze_unet(self):
        def assert_frozen(module):
            for p in module.parameters():
                assert not p.requires_grad

        def assert_unfrozen(module):
            for p in module.parameters():
                assert p.requires_grad

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = UNetControlNetXSModel(**init_dict)
        model.freeze_unet_params()

        # # check unet
        # everything expect down,mid,up blocks
        modules_from_unet = [
            model.base_time_embedding,
            model.base_conv_in,
            model.base_conv_norm_out,
            model.base_conv_out,
        ]
        for m in modules_from_unet:
            assert_frozen(m)

        optional_modules_from_unet = [
            model.base_add_time_proj,
            model.base_add_embedding,
        ]
        for m in optional_modules_from_unet:
            if m is not None:
                assert_frozen(m)

        # down blocks
        for i, d in enumerate(model.down_blocks):
            assert_frozen(d.base_resnets)
            if isinstance(d.base_attentions, nn.ModuleList):  # attentions can be list of Nones
                assert_frozen(d.base_attentions)
            if d.base_downsamplers is not None:
                assert_frozen(d.base_downsamplers)

        # mid block
        assert_frozen(model.mid_block.base_midblock)

        # up blocks
        for i, u in enumerate(model.up_blocks):
            assert_frozen(u.resnets)
            if isinstance(u.attentions, nn.ModuleList):  # attentions can be list of Nones
                assert_frozen(u.attentions)
            if u.upsamplers is not None:
                assert_frozen(u.upsamplers)

        # # check controlnet
        # everything expect down,mid,up blocks
        modules_from_controlnet = [
            model.controlnet_cond_embedding,
            model.ctrl_conv_in,
            model.control_to_base_for_conv_in,
        ]
        optional_modules_from_controlnet = [model.ctrl_time_embedding]

        for m in modules_from_controlnet:
            assert_unfrozen(m)
        for m in optional_modules_from_controlnet:
            if m is not None:
                assert_unfrozen(m)

        # down blocks
        for d in model.down_blocks:
            assert_unfrozen(d.ctrl_resnets)
            assert_unfrozen(d.base_to_ctrl)
            assert_unfrozen(d.ctrl_to_base)
            if isinstance(d.ctrl_attentions, nn.ModuleList):  # attentions can be list of Nones
                assert_unfrozen(d.ctrl_attentions)
            if d.ctrl_downsamplers is not None:
                assert_unfrozen(d.ctrl_downsamplers)
        # mid block
        assert_unfrozen(model.mid_block.base_to_ctrl)
        assert_unfrozen(model.mid_block.ctrl_midblock)
        assert_unfrozen(model.mid_block.ctrl_to_base)
        # up blocks
        for u in model.up_blocks:
            assert_unfrozen(u.ctrl_to_base)

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
            "ControlNetXSCrossAttnDownBlock2D",
            "ControlNetXSCrossAttnMidBlock2D",
            "ControlNetXSCrossAttnUpBlock2D",
        }

        assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    @is_flaky
    def test_forward_no_control(self):
        unet = self.get_dummy_unet()
        controlnet = self.get_dummy_controlnet_from_unet(unet)

        model = UNetControlNetXSModel.from_unet(unet, controlnet)

        unet = unet.to(torch_device)
        model = model.to(torch_device)

        input_ = self.dummy_input

        control_specific_input = ["controlnet_cond", "conditioning_scale"]
        input_for_unet = {k: v for k, v in input_.items() if k not in control_specific_input}

        with torch.no_grad():
            unet_output = unet(**input_for_unet).sample.cpu()
            unet_controlnet_output = model(**input_, apply_control=False).sample.cpu()

        assert np.abs(unet_output.flatten() - unet_controlnet_output.flatten()).max() < 3e-4

    def test_time_embedding_mixing(self):
        unet = self.get_dummy_unet()
        controlnet = self.get_dummy_controlnet_from_unet(unet)
        controlnet_mix_time = self.get_dummy_controlnet_from_unet(
            unet, time_embedding_mix=0.5, learn_time_embedding=True
        )

        model = UNetControlNetXSModel.from_unet(unet, controlnet)
        model_mix_time = UNetControlNetXSModel.from_unet(unet, controlnet_mix_time)

        unet = unet.to(torch_device)
        model = model.to(torch_device)
        model_mix_time = model_mix_time.to(torch_device)

        input_ = self.dummy_input

        with torch.no_grad():
            output = model(**input_).sample
            output_mix_time = model_mix_time(**input_).sample

        assert output.shape == output_mix_time.shape

    def test_forward_with_norm_groups(self):
        # UNetControlNetXSModel currently only supports StableDiffusion and StableDiffusion-XL, both of which have norm_num_groups fixed at 32. So we don't need to test different values for norm_num_groups.
        pass
