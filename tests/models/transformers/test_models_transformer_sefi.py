# coding=utf-8
# Copyright 2026 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch

from diffusers import SeFiTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin


enable_full_determinism()


class SeFiTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return SeFiTransformer2DModel

    @property
    def output_shape(self):
        return (16, 8)

    @property
    def input_shape(self):
        return (16, 8)

    @property
    def main_input_name(self):
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "patch_size": 1,
            "in_channels": 8,
            "out_channels": 8,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 16,
            "timestep_guidance_channels": 16,
            "axes_dims_rope": [2, 2, 2, 2],
        }

    def get_dummy_inputs(self, height: int = 4, width: int = 4, batch_size: int = 1):
        sequence_length = 8
        hidden_states = randn_tensor((batch_size, height * width, 8), generator=self.generator, device=torch_device)
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, 16), generator=self.generator, device=torch_device
        )

        image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(sequence_length)
        )
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep_sem = torch.tensor([1.0], device=torch_device).expand(batch_size)
        timestep_tex = torch.tensor([0.9], device=torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep_sem": timestep_sem,
            "timestep_tex": timestep_tex,
        }


class TestSeFiTransformer(SeFiTransformerTesterConfig, ModelTesterMixin):
    pass
