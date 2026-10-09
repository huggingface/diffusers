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

from diffusers import Kandinsky6SRTransformer3DModel
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


class Kandinsky6SRTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Kandinsky6SRTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_visual_dim": 4,
            "out_visual_dim": 8,
            "time_dim": 16,
            "patch_size": (1, 2, 2),
            "model_dim": 24,
            "ff_dim": 32,
            "num_visual_blocks": 2,
            "axes_dims": (4, 4, 4),
        }

    def _build_dummy_inputs(self, batch_size: int, num_frames: int, height: int, width: int) -> dict:
        # The input concatenates the noisy latent, the anchor latent and the anchor mask
        num_input_channels = 2 * self.get_init_dict()["in_visual_dim"] + 1
        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames, height, width, num_input_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), generator=self.generator).float().to(torch_device),
        }

    def get_dummy_inputs(self) -> dict:
        return self._build_dummy_inputs(batch_size=1, num_frames=2, height=16, width=16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (2, 16, 16, 2 * 4 + 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (2, 16, 16, 8)


class TestKandinsky6SRTransformerModel(Kandinsky6SRTransformerTesterConfig, ModelTesterMixin):
    pass


class TestKandinsky6SRTransformerMemory(Kandinsky6SRTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestKandinsky6SRTransformerTorchCompile(Kandinsky6SRTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(16, 16), (16, 32), (32, 32)]

    def get_dummy_inputs(self, height: int = 16, width: int = 16) -> dict:
        return self._build_dummy_inputs(batch_size=1, num_frames=2, height=height, width=width)


@pytest.mark.skipif(torch_device == "cpu", reason="FlexAttention does not support backward on CPU.")
class TestKandinsky6SRTransformerTraining(Kandinsky6SRTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestKandinsky6SRTransformerAttention(Kandinsky6SRTransformerTesterConfig, AttentionTesterMixin):
    def test_fuse_unfuse_qkv_projections(self, *args, **kwargs):
        pytest.skip(
            "Kandinsky6SRAttention names its projections to_query/to_key/to_value/out_layer (matching the "
            "Kandinsky 5 layout) rather than to_q/to_k/to_v/to_out, which "
            "AttentionModuleMixin.fuse_projections hardcodes."
        )
