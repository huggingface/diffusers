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
import torch.nn.functional as F

from diffusers import Lumina2Transformer2DModel
from diffusers.models.attention import LuminaFeedForward
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    LoraTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Lumina2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Lumina2Transformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 16,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 24,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "num_attention_heads": 3,
            "num_kv_heads": 1,
            "multiple_of": 2,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "scaling_factor": 1.0,
            "axes_dim_rope": (4, 2, 2),
            "axes_lens": (128, 128, 128),
            "cap_feat_dim": 32,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 16
        embedding_dim = 32
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.rand(size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.bool, device=torch_device),
        }


class TestLumina2Transformer(Lumina2TransformerTesterConfig, ModelTesterMixin):
    pass


class TestLumina2TransformerTraining(Lumina2TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Lumina2Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestLumina2TransformerLoRA(Lumina2TransformerTesterConfig, LoraTesterMixin):
    pass


class TestLumina2Transformer2DSingleFile(Lumina2TransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors"

    @property
    def pretrained_model_name_or_path(self):
        return "Alpha-VLLM/Lumina-Image-2.0"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}


class TestLuminaFeedForward:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    # CPU and CUDA only: their fp16 / bf16 SiLU kernels compute in float32; other backends may differ by rounding.
    @pytest.mark.parametrize("device", sorted({"cpu", torch_device} & {"cpu", "cuda"}))
    def test_silu_matches_float32_reference(self, device, dtype):
        """The SiLU output equals SiLU computed on the input upcast to float32 and cast back, bit for bit."""
        torch.manual_seed(0)
        feed_forward = LuminaFeedForward(dim=32, inner_dim=64, multiple_of=32).to(device, dtype)
        # 2 x 8 x 64 = 1024 hidden elements: a multiple of the CPU vector width, so no element takes the scalar
        # tail of the SiLU kernel (bit-exact on CPU).
        x = 4 * torch.randn(2, 8, 32, device=device, dtype=dtype)
        with torch.no_grad():
            out = feed_forward(x)
            hidden = F.silu(feed_forward.linear_1(x).float()).to(dtype) * feed_forward.linear_3(x)
            ref = feed_forward.linear_2(hidden)
        assert torch.equal(out, ref)
