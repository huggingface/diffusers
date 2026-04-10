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

import os

import pytest
import torch

from diffusers import ErnieImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


# Ernie-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations.
# Cannot use enable_full_determinism() which sets it to True.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


class ErnieImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return ErnieImageTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (16, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (16, 16, 16)

    @property
    def model_split_percents(self) -> list:
        # We override the items here because the transformer under consideration is small.
        return [0.9, 0.9, 0.9]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "hidden_size": 16,
            "num_attention_heads": 1,
            "num_layers": 1,
            "ffn_hidden_size": 16,
            "in_channels": 16,
            "out_channels": 16,
            "patch_size": 1,
            "text_in_dim": 16,
            "rope_theta": 256,
            "rope_axes_dim": (8, 4, 4),
            "eps": 1e-6,
            "qk_layernorm": True,
        }

    def get_dummy_inputs(self, height: int = 16, width: int = 16, batch_size: int = 1) -> dict:
        num_channels = 16  # in_channels
        sequence_length = 16
        text_in_dim = 16  # text_in_dim

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0] * batch_size, device=torch_device),
            "text_bth": randn_tensor(
                (batch_size, sequence_length, text_in_dim), generator=self.generator, device=torch_device
            ),
            "text_lens": torch.tensor([sequence_length] * batch_size, device=torch_device),
        }


class TestErnieImageTransformer(ErnieImageTransformerTesterConfig, ModelTesterMixin):
    pass


class TestErnieImageTransformerTraining(ErnieImageTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"ErnieImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestErnieImageTransformerCompile(ErnieImageTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    @pytest.mark.skip(
        reason="The repeated block in this model is ErnieImageSharedAdaLNBlock. As a consequence of this, "
        "the inputs recorded for the block would vary during compilation and full compilation with "
        "fullgraph=True would trigger recompilation."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        super().test_torch_compile_recompilation_and_graph_break()

    @pytest.mark.skip(reason="Fullgraph AoT is broken.")
    def test_compile_works_with_aot(self, tmp_path):
        super().test_compile_works_with_aot(tmp_path)

    @pytest.mark.skip(reason="Fullgraph is broken.")
    def test_compile_on_different_shapes(self):
        super().test_compile_on_different_shapes()
