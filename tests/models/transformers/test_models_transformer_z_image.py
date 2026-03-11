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

from diffusers import ZImageTransformer2DModel

from ...testing_utils import IS_GITHUB_ACTIONS, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
# Cannot use enable_full_determinism() which sets it to True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


pytestmark = pytest.mark.skipif(
    IS_GITHUB_ACTIONS,
    reason="Skipping test-suite inside the CI because the model has `torch.empty()` inside of it during init and we don't have a clear way to override it in the modeling tests.",
)


class ZImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return ZImageTransformer2DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 32, 32)

    @property
    def model_split_percents(self) -> list:
        return [0.9, 0.9, 0.9]

    @property
    def main_input_name(self) -> str:
        return "x"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool | float]:
        return {
            "all_patch_size": (2,),
            "all_f_patch_size": (1,),
            "in_channels": 16,
            "dim": 16,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 1,
            "n_kv_heads": 2,
            "qk_norm": True,
            "cap_feat_dim": 16,
            "rope_theta": 256.0,
            "t_scale": 1000.0,
            "axes_dims": [8, 4, 4],
            "axes_lens": [256, 32, 32],
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor | list]:
        batch_size = 1
        num_channels = 16
        embedding_dim = 16
        sequence_length = 16
        height = 16
        width = 16

        hidden_states = [torch.randn((num_channels, 1, height, width)).to(torch_device) for _ in range(batch_size)]
        encoder_hidden_states = [
            torch.randn((sequence_length, embedding_dim)).to(torch_device) for _ in range(batch_size)
        ]
        timestep = torch.tensor([0.0]).to(torch_device)

        return {"x": hidden_states, "cap_feats": encoder_hidden_states, "t": timestep}


class TestZImageTransformer(ZImageTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for Z-Image Transformer."""

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_outputs_equivalence(self, atol=1e-5, rtol=0):
        pass


class TestZImageTransformerMemory(ZImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Z-Image Transformer."""

    @pytest.mark.skip("Test will pass if we change to deterministic values instead of empty in the DiT.")
    def test_group_offloading(self, record_stream, atol=1e-5, rtol=0):
        pass

    @pytest.mark.skip("Test will pass if we change to deterministic values instead of empty in the DiT.")
    def test_group_offloading_with_disk(self, tmp_path, record_stream, offload_type, atol=1e-5, rtol=0):
        pass

    @pytest.mark.skip(
        "Test needs to be revisited. Ensure `x_pad_token` and `cap_pad_token` are cast to the same dtype as the destination tensor before they are assigned to the padding indices."
    )
    def test_layerwise_casting_training(self):
        pass


class TestZImageTransformerTraining(ZImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Z-Image Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"ZImageTransformer2DModel"})

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_training(self):
        pass

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_training_with_ema(self):
        pass

    @pytest.mark.skip("Test is not supported for handling main inputs that are lists.")
    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        pass


class TestZImageTransformerCompile(ZImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Z-Image Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 16, width: int = 16) -> dict[str, torch.Tensor | list]:
        batch_size = 1
        num_channels = 16
        embedding_dim = 16
        sequence_length = 16

        hidden_states = [torch.randn((num_channels, 1, height, width)).to(torch_device) for _ in range(batch_size)]
        encoder_hidden_states = [
            torch.randn((sequence_length, embedding_dim)).to(torch_device) for _ in range(batch_size)
        ]
        timestep = torch.tensor([0.0]).to(torch_device)

        return {"x": hidden_states, "cap_feats": encoder_hidden_states, "t": timestep}

    @pytest.mark.skip(
        "The repeated block in this model is ZImageTransformerBlock, which is used for noise_refiner, context_refiner, and layers. The inputs recorded for the block would vary during compilation and full compilation with fullgraph=True would trigger recompilation at least thrice."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    @pytest.mark.skip("Fullgraph AoT is broken")
    def test_compile_works_with_aot(self, tmp_path):
        pass

    @pytest.mark.skip("Fullgraph is broken")
    def test_compile_on_different_shapes(self):
        pass
