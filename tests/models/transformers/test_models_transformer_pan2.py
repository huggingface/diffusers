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

from diffusers import PAN2Transformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class PAN2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return PAN2Transformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "wuqing157/tiny-pan2-modular-pipe"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (9, 2, 16, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | tuple[int, ...] | float]:
        return {
            "patch_size": (1, 2, 2),
            "in_channels": 9,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "mlp_ratio": 2.0,
            "context_mlp_ratio": 1.0,
            "refiner_mlp_ratio": 1.0,
            "text_embed_dim": 16,
            "rope_axes_dim": (4, 6, 6),
        }

    def get_dummy_inputs(self, height: int = 16, width: int = 16, batch_size: int = 1) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor(
                (batch_size, 9, 2, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, 12, 16), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.full((batch_size,), 500.0, device=torch_device),
        }


class TestPAN2Transformer(PAN2TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the PAN2 transformer."""


class TestPAN2TransformerMemory(PAN2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the PAN2 transformer."""


class TestPAN2TransformerTraining(PAN2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the PAN2 transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"PAN2Transformer3DModel"})


class TestPAN2TransformerAttention(PAN2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the PAN2 transformer."""


class TestPAN2TransformerCompile(PAN2TransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the PAN2 transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(16, 16), (16, 32), (32, 32)]


class TestPAN2TransformerContextParallel(PAN2TransformerTesterConfig, ContextParallelTesterMixin):
    """Context parallel inference tests for the PAN2 transformer."""
