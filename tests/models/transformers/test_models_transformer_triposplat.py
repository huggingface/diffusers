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

import torch

from diffusers import TripoSplatTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class TripoSplatTransformerTesterConfig:
    main_input_name = "hidden_states"

    @property
    def model_class(self):
        return TripoSplatTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "q_token_length": 16,
            "in_channels": 16,
            "out_channels": 16,
            "cam_channels": 5,
            "model_channels": 32,
            "cond_channels": 32,
            "cond2_channels": 8,
            "num_refiner_blocks": 1,
            "num_blocks": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 16, 16), generator=self.generator, device=torch_device),
            "timestep": torch.tensor([500.0], device=torch_device),
            "encoder_hidden_states": randn_tensor((1, 9, 32), generator=self.generator, device=torch_device),
            "image_latents": randn_tensor((1, 9, 8), generator=self.generator, device=torch_device),
            "camera_latents": randn_tensor((1, 1, 5), generator=self.generator, device=torch_device),
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 16)


class TestTripoSplatTransformerModel(TripoSplatTransformerTesterConfig, ModelTesterMixin):
    pass


class TestTripoSplatTransformerMemory(TripoSplatTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestTripoSplatTransformerTorchCompile(TripoSplatTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 16, 16), generator=self.generator, device=torch_device),
            "timestep": torch.tensor([500.0], device=torch_device),
            "encoder_hidden_states": randn_tensor(
                (1, height * width, 32), generator=self.generator, device=torch_device
            ),
            "image_latents": randn_tensor((1, height * width, 8), generator=self.generator, device=torch_device),
            "camera_latents": randn_tensor((1, 1, 5), generator=self.generator, device=torch_device),
        }


class TestTripoSplatTransformerTraining(TripoSplatTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestTripoSplatTransformerAttention(TripoSplatTransformerTesterConfig, AttentionTesterMixin):
    pass
