# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers import JoyVideoEditTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class JoyVideoEditTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return JoyVideoEditTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (8, 2, 4, 4)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (8, 2, 4, 4)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": [1, 1, 1],
            "in_channels": 8,
            "out_channels": 8,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "text_dim": 16,
            "num_layers": 2,
            "rope_dim_list": [4, 6, 6],
            "theta": 256,
            "chunk_size": 1,
            "local_window_size": 3,
            "global_sink_chunk": True,
            "source_id_rope_dim": 4,
            "source_id_rope_theta": 256.0,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_frames, height, width = 2, 4, 4
        hidden_states = randn_tensor(
            (batch_size, 8, num_frames, height, width), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor((batch_size, 8, 16), generator=self.generator, device=torch_device)
        encoder_hidden_states_mask = torch.ones(batch_size, 8, dtype=torch.bool, device=torch_device)
        encoder_hidden_states_mask[:, -1] = False
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
        }


class TestJoyVideoEditTransformerModel(JoyVideoEditTransformerTesterConfig, ModelTesterMixin):
    def test_invalid_self_attn_input_mode_raises(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with pytest.raises(ValueError, match="Unsupported self-attention input mode"):
            model(**self.get_dummy_inputs(), self_attn_input_mode="invalid")

    def test_temporal_ids_are_applied_per_batch_element(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs["hidden_states"] = inputs["hidden_states"].expand(2, -1, -1, -1, -1).clone()
        inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].expand(2, -1, -1).clone()
        inputs["encoder_hidden_states_mask"] = inputs["encoder_hidden_states_mask"].expand(2, -1).clone()
        inputs["timestep"] = inputs["timestep"].expand(2).clone()
        inputs["current_temporal_ids"] = torch.tensor([[0, 1], [0, 3]], device=torch_device)

        with torch.no_grad():
            batch_output = model(**inputs).sample
            first_output = model(
                **{key: value[:1] for key, value in inputs.items() if key != "current_temporal_ids"},
                current_temporal_ids=inputs["current_temporal_ids"][:1],
            ).sample
            second_output = model(
                **{key: value[1:] for key, value in inputs.items() if key != "current_temporal_ids"},
                current_temporal_ids=inputs["current_temporal_ids"][1:],
            ).sample

        torch.testing.assert_close(batch_output[:1], first_output)
        torch.testing.assert_close(batch_output[1:], second_output)


class TestJoyVideoEditTransformerMemory(JoyVideoEditTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestJoyVideoEditTransformerTraining(JoyVideoEditTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"JoyVideoEditTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestJoyVideoEditTransformerAttention(JoyVideoEditTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestJoyVideoEditTransformerCompile(JoyVideoEditTransformerTesterConfig, TorchCompileTesterMixin):
    pass
