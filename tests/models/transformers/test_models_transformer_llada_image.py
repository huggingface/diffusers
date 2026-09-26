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

from diffusers import (
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageTransformer2DModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


def _flatten_list_output(output: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([sample.flatten() for sample in output])


class LLaDAImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LLaDAImageTransformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return "inclusionAI/LLaDA-Image"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    @property
    def main_input_name(self) -> str:
        return "x"

    @property
    def model_split_percents(self) -> list[float]:
        return [0.9, 0.9, 0.9]

    def get_init_dict(self) -> dict[str, int | list[int]]:
        # __init__ parameters:
        #   all_patch_size: tuple[int, Ellipsis] = <complex>
        #   all_f_patch_size: tuple[int, Ellipsis] = <complex>
        #   in_channels: int = 128
        #   dim: int = 3840
        #   n_layers: int = 30
        #   n_refiner_layers: int = 2
        #   n_heads: int = 30
        #   norm_eps: float = 1e-05
        #   qk_norm: bool = True
        #   cap_feat_dim: int = 2560
        #   semantic_feat_dim: int = 4096
        #   rope_theta: float = 256.0
        #   t_scale: float = 1000.0
        #   axes_dims: tuple[int, Ellipsis] = <complex>
        #   axes_lens: tuple[int, Ellipsis] = <complex>
        return {
            "in_channels": 8,
            "dim": 32,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 2,
            "cap_feat_dim": 24,
            "semantic_feat_dim": 20,
            "axes_dims": (4, 6, 6),
            "axes_lens": (2048, 32, 32),
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        # forward() parameters:
        #   x: list[torch.Tensor]
        #   t: torch.Tensor
        #   cap_feats: list[torch.Tensor] | None
        #   glm_cap_feats: list[torch.Tensor] | None
        #   source_latents: list[torch.Tensor] | None
        #   patch_size: int = 1
        #   f_patch_size: int = 1
        #   return_dict: bool = True
        return self.get_inputs_with_shapes(4, 4)

    def get_inputs_with_shapes(self, height: int, width: int) -> dict[str, torch.Tensor]:
        return {
            "x": [
                randn_tensor((8, 1, height, width), generator=self.generator, device=torch_device) for _ in range(2)
            ],
            "t": torch.tensor([0.8, 0.8], device=torch_device),
            "cap_feats": [
                randn_tensor((length, 24), generator=self.generator, device=torch_device) for length in (5, 7)
            ],
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (2, 8, 1, 4, 4)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (8, 1, 4, 4)


class TestLLaDAImageTransformerModel(LLaDAImageTransformerTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_determinism(self, atol=1e-5, rtol=0):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        first = _flatten_list_output(model(**inputs, return_dict=False)[0])
        second = _flatten_list_output(model(**inputs, return_dict=False)[0])
        mask = ~(torch.isnan(first) | torch.isnan(second))
        assert_tensors_close(first[mask], second[mask], atol=atol, rtol=rtol)

    @pytest.mark.skip("The model returns a list so it can preserve per-sample spatial shapes.")
    def test_outputs_equivalence(self, atol=1e-5, rtol=0):
        pass


class TestLLaDAImageTransformerMemory(LLaDAImageTransformerTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip("The shared training test does not support list-valued main inputs.")
    def test_layerwise_casting_training(self):
        pass


class TestLLaDAImageTransformerTorchCompile(LLaDAImageTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        return self.get_inputs_with_shapes(height, width)

    def test_torch_compile_repeated_blocks(self):
        # The same block class is used by two refiners and the denoising stack with different attention processors.
        super().test_torch_compile_repeated_blocks(recompile_limit=3)

    @pytest.mark.skip("AOTInductor package loading does not support this model's list-valued inputs.")
    def test_compile_works_with_aot(self, tmp_path):
        pass


class TestLLaDAImageTransformerTraining(LLaDAImageTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"LLaDAImageTransformer2DModel"})

    @pytest.mark.skip("The shared training test does not support list-valued model outputs.")
    def test_training(self):
        pass

    @pytest.mark.skip("The shared training test does not support list-valued model outputs.")
    def test_training_with_ema(self):
        pass

    @pytest.mark.skip("The shared mixed-precision test does not support list-valued model outputs.")
    def test_mixed_precision_training(self):
        pass

    @pytest.mark.skip("The shared gradient comparison does not support list-valued model outputs.")
    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        pass


class TestLLaDAImageTransformerAttention(LLaDAImageTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestLLaDAImageAuxiliaryModels:
    def test_queryformer(self):
        torch.manual_seed(0)
        model = LLaDAImageQueryFormerModel(
            num_queries=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
        ).to(torch_device)
        hidden_states = torch.randn(2, 5, 16, device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=torch_device)
        output = model(hidden_states, attention_mask).query_embeds
        assert output.shape == (2, 4, 16)

    def test_text_projection(self):
        torch.manual_seed(0)
        model = LLaDAImageTextProjectionModel(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            projection_dim=24,
        ).to(torch_device)
        output = model(torch.randn(2, 7, 16, device=torch_device)).hidden_states
        assert output.shape == (2, 7, 24)

    def test_sigvq_image_and_tokens(self):
        torch.manual_seed(0)
        model = LLaDAImageSigVQModel(
            image_size=32,
            patch_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            codebook_size=32,
            codebook_embed_dim=8,
            semantic_embed_dim=20,
        ).to(torch_device)
        image_output = model(pixel_values=torch.randn(2, 3, 32, 32, device=torch_device))
        token_output = model(token_ids=image_output.token_ids)

        assert image_output.semantic_features.shape == (2, 16, 20)
        assert image_output.token_ids.shape == (2, 16)
        torch.testing.assert_close(token_output.semantic_features, image_output.semantic_features)
