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

import torch
from torch import nn

from diffusers.models import DualTransformer2DModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D


class CapturingTransformer(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.calls = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        encoder_attention_mask=None,
        cross_attention_kwargs=None,
        return_dict=True,
    ):
        self.calls.append(
            {
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timestep,
                "attention_mask": attention_mask,
                "encoder_attention_mask": encoder_attention_mask,
                "cross_attention_kwargs": cross_attention_kwargs,
                "return_dict": return_dict,
            }
        )
        return (hidden_states + self.delta,)


class DualTransformer2DModelTests(unittest.TestCase):
    def get_model_with_capturing_transformers(self):
        model = DualTransformer2DModel(
            num_attention_heads=1,
            attention_head_dim=4,
            in_channels=4,
            num_layers=1,
            norm_num_groups=1,
            cross_attention_dim=4,
        )
        transformer_0 = CapturingTransformer(delta=4)
        transformer_1 = CapturingTransformer(delta=2)
        model.transformers = nn.ModuleList([transformer_0, transformer_1])
        model.condition_lengths = [2, 3]
        model.transformer_index_for_condition = [1, 0]
        return model, transformer_0, transformer_1

    def check_mask_routing(self, encoder_attention_mask):
        model, transformer_0, transformer_1 = self.get_model_with_capturing_transformers()

        hidden_states = torch.randn(1, 4, 2, 2)
        encoder_hidden_states = torch.randn(1, 5, 4)
        attention_mask = torch.ones(1, 4)
        timestep = torch.tensor([1])
        cross_attention_kwargs = {"foo": "bar"}

        output = model(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        assert torch.equal(output.sample, hidden_states + 3)

        assert len(transformer_1.calls) == 1
        first_condition_call = transformer_1.calls[0]
        assert first_condition_call["attention_mask"] is attention_mask
        assert first_condition_call["timestep"] is timestep
        assert first_condition_call["cross_attention_kwargs"] is cross_attention_kwargs
        assert first_condition_call["return_dict"] is False
        assert torch.equal(first_condition_call["encoder_hidden_states"], encoder_hidden_states[:, :2])
        assert torch.equal(first_condition_call["encoder_attention_mask"], encoder_attention_mask[..., :2])

        assert len(transformer_0.calls) == 1
        second_condition_call = transformer_0.calls[0]
        assert second_condition_call["attention_mask"] is attention_mask
        assert second_condition_call["timestep"] is timestep
        assert second_condition_call["cross_attention_kwargs"] is cross_attention_kwargs
        assert second_condition_call["return_dict"] is False
        assert torch.equal(second_condition_call["encoder_hidden_states"], encoder_hidden_states[:, 2:5])
        assert torch.equal(second_condition_call["encoder_attention_mask"], encoder_attention_mask[..., 2:5])

    def test_forward_passes_attention_masks_to_child_transformers(self):
        self.check_mask_routing(torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]]))
        self.check_mask_routing(torch.tensor([[[0.0, 0.0, -10000.0, 0.0, -10000.0]]]))

    def test_forward_tuple_output(self):
        model, _, _ = self.get_model_with_capturing_transformers()

        hidden_states = torch.randn(1, 4, 2, 2)
        output = model(
            hidden_states,
            encoder_hidden_states=torch.randn(1, 5, 4),
            return_dict=False,
        )

        assert isinstance(output, tuple)
        assert torch.equal(output[0], hidden_states + 3)

    def test_cross_attn_down_block_dual_cross_attention_accepts_encoder_attention_mask(self):
        block = CrossAttnDownBlock2D(
            in_channels=4,
            out_channels=4,
            temb_channels=8,
            num_layers=1,
            transformer_layers_per_block=1,
            num_attention_heads=1,
            cross_attention_dim=8,
            dual_cross_attention=True,
            resnet_groups=1,
            add_downsample=False,
        )
        block.eval()

        with torch.no_grad():
            output = block(
                torch.randn(1, 4, 4, 4),
                temb=torch.randn(1, 8),
                encoder_hidden_states=torch.randn(1, 77 + 257, 8),
                attention_mask=torch.ones(1, 16),
                encoder_attention_mask=torch.ones(1, 77 + 257),
            )

        assert output[0].shape == (1, 4, 4, 4)

    def test_top_level_import(self):
        from diffusers import DualTransformer2DModel as TopLevelDualTransformer2DModel
        from diffusers.models import DualTransformer2DModel as ModelsDualTransformer2DModel

        assert TopLevelDualTransformer2DModel is ModelsDualTransformer2DModel
