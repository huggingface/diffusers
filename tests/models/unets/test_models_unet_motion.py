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
import os

import torch

from diffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class UNetMotionModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return UNetMotionModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (4, 4, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "block_out_channels": (16, 32),
            "norm_num_groups": 16,
            "down_block_types": ("CrossAttnDownBlockMotion", "DownBlockMotion"),
            "up_block_types": ("UpBlockMotion", "CrossAttnUpBlockMotion"),
            "cross_attention_dim": 16,
            "num_attention_heads": 2,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 16,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (16, 16)
        noise = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        timestep = torch.tensor([10], device=torch_device)
        encoder_hidden_states = randn_tensor(
            (batch_size * num_frames, 4, 16), generator=self.generator, device=torch_device
        )
        return {"sample": noise, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states}


class TestUNetMotionModel(UNetMotionModelTesterConfig, ModelTesterMixin):
    def test_from_unet2d(self):
        torch.manual_seed(0)
        unet2d = UNet2DConditionModel()

        torch.manual_seed(1)
        model = self.model_class.from_unet2d(unet2d)
        model_state_dict = model.state_dict()

        for param_name, param_value in unet2d.named_parameters():
            assert torch.equal(model_state_dict[param_name], param_value)

    def test_freeze_unet2d(self):
        model = self.model_class(**self.get_init_dict())
        model.freeze_unet2d_params()

        for param_name, param_value in model.named_parameters():
            if "motion_modules" not in param_name:
                assert not param_value.requires_grad
            else:
                assert param_value.requires_grad

    def test_loading_motion_adapter(self):
        model = self.model_class()
        adapter = MotionAdapter()
        model.load_motion_modules(adapter)

        for idx, down_block in enumerate(model.down_blocks):
            adapter_state_dict = adapter.down_blocks[idx].motion_modules.state_dict()
            for param_name, param_value in down_block.motion_modules.named_parameters():
                assert torch.equal(adapter_state_dict[param_name], param_value)

        for idx, up_block in enumerate(model.up_blocks):
            adapter_state_dict = adapter.up_blocks[idx].motion_modules.state_dict()
            for param_name, param_value in up_block.motion_modules.named_parameters():
                assert torch.equal(adapter_state_dict[param_name], param_value)

        mid_block_adapter_state_dict = adapter.mid_block.motion_modules.state_dict()
        for param_name, param_value in model.mid_block.motion_modules.named_parameters():
            assert torch.equal(mid_block_adapter_state_dict[param_name], param_value)

    def test_saving_motion_modules(self, tmp_path):
        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        model.save_motion_modules(tmp_path)
        assert os.path.isfile(os.path.join(tmp_path, "diffusion_pytorch_model.safetensors"))

        adapter_loaded = MotionAdapter.from_pretrained(tmp_path)
        torch.manual_seed(0)
        model_loaded = self.model_class(**init_dict)
        model_loaded.load_motion_modules(adapter_loaded)
        model_loaded.to(torch_device)

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())[0]
            output_loaded = model_loaded(**self.get_dummy_inputs())[0]

        assert (output - output_loaded).abs().max().item() <= 1e-4, "Models give different forward passes"

    def test_feed_forward_chunking(self):
        init_dict = self.get_init_dict()
        init_dict["block_out_channels"] = (32, 64)
        init_dict["norm_num_groups"] = 32
        model = self.model_class(**init_dict).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())[0]

        model.enable_forward_chunking()
        with torch.no_grad():
            output_2 = model(**self.get_dummy_inputs())[0]

        assert output.shape == output_2.shape, "Shape doesn't match"
        assert (output - output_2).abs().max() < 1e-2

    def test_pickle(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device)

        with torch.no_grad():
            sample = model(**self.get_dummy_inputs()).sample

        sample_copy = copy.copy(sample)
        assert (sample - sample_copy).abs().max() < 1e-4

    def test_forward_with_norm_groups(self):
        init_dict = self.get_init_dict()
        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32)
        model = self.model_class(**init_dict).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample

        assert output.shape == self.get_dummy_inputs()["sample"].shape, "Input and output shapes do not match"

    def test_asymmetric_motion_model(self):
        init_dict = self.get_init_dict()
        init_dict["layers_per_block"] = (2, 3)
        init_dict["transformer_layers_per_block"] = ((1, 2), (3, 4, 5))
        init_dict["reverse_transformer_layers_per_block"] = ((7, 6, 7, 4), (4, 2, 2))
        init_dict["temporal_transformer_layers_per_block"] = ((2, 5), (2, 3, 5))
        init_dict["reverse_temporal_transformer_layers_per_block"] = ((5, 4, 3, 4), (3, 2, 2))
        init_dict["num_attention_heads"] = (2, 4)
        init_dict["motion_num_attention_heads"] = (4, 4)
        init_dict["reverse_motion_num_attention_heads"] = (2, 2)
        init_dict["use_motion_mid_block"] = True
        init_dict["mid_block_layers"] = 2
        init_dict["transformer_layers_per_mid_block"] = (1, 5)
        init_dict["temporal_transformer_layers_per_mid_block"] = (2, 4)
        model = self.model_class(**init_dict).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample

        assert output.shape == self.get_dummy_inputs()["sample"].shape, "Input and output shapes do not match"


class TestUNetMotionModelTraining(UNetMotionModelTesterConfig, TrainingTesterMixin):
    """Training tests for UNetMotionModel."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "CrossAttnUpBlockMotion",
            "CrossAttnDownBlockMotion",
            "UNetMidBlockCrossAttnMotion",
            "UpBlockMotion",
            "Transformer2DModel",
            "DownBlockMotion",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestUNetMotionModelMemory(UNetMotionModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for UNetMotionModel."""


class TestUNetMotionModelAttention(UNetMotionModelTesterConfig, AttentionTesterMixin):
    """Attention processor tests for UNetMotionModel."""
