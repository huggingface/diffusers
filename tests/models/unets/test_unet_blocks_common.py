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
from typing import Tuple

import torch

from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    floats_tensor,
    require_torch,
    require_torch_accelerator_with_training,
    torch_all_close,
    torch_device,
)


@require_torch
class UNetBlockTesterMixin:
    @property
    def dummy_input(self):
        return self.get_dummy_input()

    @property
    def output_shape(self):
        if self.block_type == "down":
            return (4, 32, 16, 16)
        elif self.block_type == "mid":
            return (4, 32, 32, 32)
        elif self.block_type == "up":
            return (4, 32, 64, 64)

        raise ValueError(f"'{self.block_type}' is not a supported block_type. Set it to 'up', 'mid', or 'down'.")

    def get_dummy_input(
        self,
        include_temb=True,
        include_res_hidden_states_tuple=False,
        include_encoder_hidden_states=False,
        include_skip_sample=False,
    ):
        batch_size = 4
        num_channels = 32
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        device = torch.device(torch_device)
        shape = (batch_size, num_channels) + sizes
        hidden_states = randn_tensor(shape, generator=generator, device=device)
        dummy_input = {"hidden_states": hidden_states}

        if include_temb:
            temb_channels = 128
            dummy_input["temb"] = randn_tensor((batch_size, temb_channels), generator=generator, device=device)

        if include_res_hidden_states_tuple:
            generator_1 = torch.manual_seed(1)
            dummy_input["res_hidden_states_tuple"] = (randn_tensor(shape, generator=generator_1, device=device),)

        if include_encoder_hidden_states:
            dummy_input["encoder_hidden_states"] = floats_tensor((batch_size, 32, 32)).to(torch_device)

        if include_skip_sample:
            dummy_input["skip_sample"] = randn_tensor(((batch_size, 3) + sizes), generator=generator, device=device)

        return dummy_input

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 32,
            "out_channels": 32,
            "temb_channels": 128,
        }
        if self.block_type == "up":
            init_dict["prev_output_channel"] = 32

        if self.block_type == "mid":
            init_dict.pop("out_channels")

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self, expected_slice):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        unet_block = self.block_class(**init_dict)
        unet_block.to(torch_device)
        unet_block.eval()

        with torch.no_grad():
            output = unet_block(**inputs_dict)

        if isinstance(output, Tuple):
            output = output[0]

        self.assertEqual(output.shape, self.output_shape)

        output_slice = output[0, -1, -3:, -3:]
        expected_slice = torch.tensor(expected_slice).to(torch_device)
        assert torch_all_close(output_slice.flatten(), expected_slice, atol=5e-3)

    @require_torch_accelerator_with_training
    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.block_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict)

        if isinstance(output, Tuple):
            output = output[0]

        device = torch.device(torch_device)
        noise = randn_tensor(output.shape, device=device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
