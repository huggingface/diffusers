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

import pytest
import torch

from diffusers import PixArtTransformer2DModel, Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, slow, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class PixArtTransformer2DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return PixArtTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (4, 8, 8)

    @property
    def output_shape(self) -> tuple:
        return (8, 8, 8)

    @property
    def model_split_percents(self) -> list:
        # We override the items here because the transformer under consideration is small.
        return [0.7, 0.6, 0.6]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 8,
            "num_layers": 1,
            "patch_size": 2,
            "attention_head_dim": 2,
            "num_attention_heads": 2,
            "in_channels": 4,
            "cross_attention_dim": 8,
            "out_channels": 8,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 8,
            "norm_type": "ada_norm_single",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "use_additional_conditions": False,
            "caption_channels": None,
        }

    def get_dummy_inputs(self, batch_size: int = 4) -> dict[str, torch.Tensor]:
        in_channels = 4
        sample_size = 8
        scheduler_num_train_steps = 1000
        cross_attention_dim = 8
        seq_len = 8

        return {
            "hidden_states": randn_tensor(
                (batch_size, in_channels, sample_size, sample_size), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, scheduler_num_train_steps, size=(batch_size,), generator=self.generator).to(
                torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, seq_len, cross_attention_dim), generator=self.generator, device=torch_device
            ),
            "added_cond_kwargs": {"aspect_ratio": None, "resolution": None},
        }


class TestPixArtTransformer2D(PixArtTransformer2DTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")

    def test_correct_class_remapping_from_dict_config(self):
        init_dict = self.get_init_dict()
        model = Transformer2DModel.from_config(init_dict)
        assert isinstance(model, PixArtTransformer2DModel)

    def test_correct_class_remapping_from_pretrained_config(self):
        config = PixArtTransformer2DModel.load_config("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="transformer")
        model = Transformer2DModel.from_config(config)
        assert isinstance(model, PixArtTransformer2DModel)

    @slow
    def test_correct_class_remapping(self):
        model = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="transformer")
        assert isinstance(model, PixArtTransformer2DModel)


class TestPixArtTransformer2DMemory(PixArtTransformer2DTesterConfig, MemoryTesterMixin):
    pass


class TestPixArtTransformer2DAttention(PixArtTransformer2DTesterConfig, AttentionTesterMixin):
    pass


class TestPixArtTransformer2DTraining(PixArtTransformer2DTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"PixArtTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
