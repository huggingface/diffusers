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

from diffusers import LensTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    ContextParallelTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class LensTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LensTransformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return ""

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.1, 0.1, 0.1]

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 20,
            "num_attention_heads": 1,
            "inner_dim": 20,
            "enc_hidden_dim": 32,
            "axes_dims_rope": [4, 8, 8],
            "gate_mlp": True,
            "rms_norm": True,
            "multi_layer_encoder_feature": True,
            "selected_layer_index": (0,),
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict:
        height = width = 4
        num_latent_channels = 16
        text_seq_len = 8
        enc_hidden_dim = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, height * width, num_latent_channels),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": [
                randn_tensor(
                    (batch_size, text_seq_len, enc_hidden_dim),
                    generator=self.generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype).expand(batch_size),
            "img_shapes": [(1, height, width)],
        }

    @property
    def input_shape(self) -> tuple:
        return (16, 16)

    @property
    def output_shape(self) -> tuple:
        return (16, 16)


class TestLensTransformerModel(LensTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        pytest.skip("Tolerance requirements too high for meaningful test")

    def test_model_parallelism(self, tmp_path):
        pytest.skip("Tiny Lens test config does not split meaningfully across multiple GPUs")


class TestLensTransformerMemory(LensTransformerTesterConfig, MemoryTesterMixin):
    def test_layerwise_casting_memory(self):
        pytest.skip("Tiny Lens test config does not give a stable layerwise-casting memory ordering signal")

    def test_cpu_offload(self, tmp_path):
        pytest.skip("Tiny Lens test config does not split meaningfully for CPU offload")

    def test_disk_offload_without_safetensors(self, tmp_path):
        pytest.skip("Tiny Lens test config does not split meaningfully for disk offload")

    def test_disk_offload_with_safetensors(self, tmp_path):
        pytest.skip("Tiny Lens test config does not split meaningfully for disk offload")


class TestLensTransformerTorchCompile(LensTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict:
        num_latent_channels = 16
        text_seq_len = 8
        enc_hidden_dim = 32
        batch_size = 1

        return {
            "hidden_states": randn_tensor(
                (batch_size, height * width, num_latent_channels),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": [
                randn_tensor(
                    (batch_size, text_seq_len, enc_hidden_dim),
                    generator=self.generator,
                    device=torch_device,
                    dtype=self.torch_dtype,
                )
            ],
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype).expand(batch_size),
            "img_shapes": [(1, height, width)],
        }


@pytest.mark.skip(reason="Tiny Lens test config is not suitable for context-parallel coverage")
class TestLensTransformerContextParallel(LensTransformerTesterConfig, ContextParallelTesterMixin):
    pass


class TestLensTransformerTraining(LensTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestLensTransformerAttention(LensTransformerTesterConfig, AttentionTesterMixin):
    def test_fuse_unfuse_qkv_projections(self):
        pytest.skip("LensJointAttention does not use the generic to_q/to_k/to_v fusion path")

    def test_attention_processor_count_mismatch_raises_error(self):
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            pytest.skip("Model does not support setting attention processors.")

        current_processors = model.attn_processors
        if len(current_processors) <= 1:
            pytest.skip("Lens test config exposes only one attention processor.")

        return super().test_attention_processor_count_mismatch_raises_error()
