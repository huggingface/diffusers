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

import os
from unittest import mock

import pytest
import torch

from diffusers import ErnieImageTransformer2DModel
from diffusers.models.transformers.transformer_ernie_image import ErnieImageSharedAdaLNBlock
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


class TestErnieImageTransformerAdaLN(ErnieImageTransformerTesterConfig):
    def test_adaln_broadcast_matches_expanded_float32_path(self):
        """Broadcasting the [1, B, H] modulation tensors gives the same bfloat16 output, bit for bit, as expanding
        them to [S, B, H] and upcasting the activations to float32 before the products."""

        def block_forward_reference(block, x, rotary_pos_emb, temb, attention_mask=None):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                t.expand(x.shape[0], -1, -1).contiguous() for t in temb
            ]
            residual = x
            x = block.adaLN_sa_ln(x)
            x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
            attn_out = block.self_attention(
                x.permute(1, 0, 2), attention_mask=attention_mask, image_rotary_emb=rotary_pos_emb
            )
            x = residual + (gate_msa.float() * attn_out.permute(1, 0, 2).float()).to(x.dtype)
            residual = x
            x = block.adaLN_mlp_ln(x)
            x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
            return residual + (gate_mlp.float() * block.mlp(x).float()).to(x.dtype)

        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device, torch.bfloat16).eval()
        with torch.no_grad():
            # zero-initialised: gate = 0 would hide the attention and MLP outputs, and the output would be all zeros
            for linear in (model.adaLN_modulation[-1], model.final_linear):
                linear.weight.normal_(std=0.5)
                linear.bias.normal_(std=0.5)

        inputs = self.get_dummy_inputs(batch_size=2)
        inputs["hidden_states"] = inputs["hidden_states"].to(torch.bfloat16)
        inputs["text_bth"] = inputs["text_bth"].to(torch.bfloat16)
        # a different timestep per sample, so the two samples get different scale / shift / gate
        inputs["timestep"] = torch.tensor([1.0, 500.0], device=torch_device)
        inputs["text_lens"] = torch.tensor([16, 9], device=torch_device)

        with torch.no_grad():
            out = model(**inputs).sample
            with mock.patch.object(ErnieImageSharedAdaLNBlock, "forward", block_forward_reference):
                ref = model(**inputs).sample
        assert torch.equal(out, ref)


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
