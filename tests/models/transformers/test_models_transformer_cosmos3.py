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

from diffusers import Cosmos3OmniTransformer
from diffusers.models.transformers.transformer_cosmos3 import (
    Cosmos3NemotronRMSNorm,
    Cosmos3OmniTransformerOutput,
    Cosmos3PackedMoTAttention,
)
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


class Cosmos3OmniTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Cosmos3OmniTransformer

    @property
    def main_input_name(self) -> str:
        return "vision_tokens"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "head_dim": 6,
            "hidden_act": "relu2",
            "hidden_size": 12,
            "intermediate_size": 24,
            "latent_channel": 2,
            "latent_patch_size": 1,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "patch_latent_dim": 2,
            "qk_norm_for_text": False,
            "rms_norm_eps": 1e-5,
            "rope_axes_dim": [1, 1, 1],
            "rope_theta": 1e8,
            "vocab_size": 32,
        }

    def get_dummy_inputs(self, height: int = 1, width: int = 1) -> dict:
        num_vision_tokens = height * width
        sequence_length = 2 + num_vision_tokens
        vision_indexes = torch.arange(2, sequence_length, device=torch_device)

        return {
            "input_ids": torch.tensor([1, 2], device=torch_device),
            "text_indexes": torch.tensor([0, 1], device=torch_device),
            "position_ids": torch.zeros((3, sequence_length), dtype=torch.long, device=torch_device),
            "und_len": 2,
            "sequence_length": sequence_length,
            "vision_tokens": [randn_tensor((1, 2, 1, height, width), generator=self.generator, device=torch_device)],
            "vision_token_shapes": [(1, height, width)],
            "vision_sequence_indexes": vision_indexes,
            "vision_mse_loss_indexes": vision_indexes,
            "vision_timesteps": torch.ones(num_vision_tokens, device=torch_device),
            "vision_noisy_frame_indexes": [torch.tensor([0], device=torch_device)],
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 2, 1, 1, 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 2, 1, 1, 1)


class TestCosmos3OmniTransformerModel(Cosmos3OmniTransformerTesterConfig, ModelTesterMixin):
    def test_output_format(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
            output_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        assert isinstance(output, Cosmos3OmniTransformerOutput)
        assert output.sample[0].shape == self.output_shape
        assert output.sound is None
        assert output.action is None
        torch.testing.assert_close(output.sample[0], output_tuple[0][0])

    def test_determinism(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            first = model(**self.get_dummy_inputs()).sample[0]
            second = model(**self.get_dummy_inputs()).sample[0]

        torch.testing.assert_close(first, second)

    def test_outputs_equivalence(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
            output_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        torch.testing.assert_close(output.sample[0], output_tuple[0][0])
        assert output.sound is output_tuple[1] is None
        assert output.action is output_tuple[2] is None

    def test_cosmos3_edge_uses_nemotron_parameter_layout(self):
        transformer = self.model_class(
            **self.get_init_dict(),
            action_dim=3,
            action_gen=True,
            num_embodiment_domains=2,
            use_und_k_norm_for_gen=True,
        )
        state_dict = transformer.state_dict()
        layer = transformer.layers[0]

        assert transformer.config.use_und_k_norm_for_gen
        assert isinstance(layer.self_attn.norm_q, torch.nn.Identity)
        assert isinstance(layer.self_attn.norm_k, torch.nn.Identity)
        assert isinstance(layer.self_attn.norm_added_q, Cosmos3NemotronRMSNorm)
        assert isinstance(layer.self_attn.norm_added_k, Cosmos3NemotronRMSNorm)
        assert isinstance(layer.input_layernorm, Cosmos3NemotronRMSNorm)
        assert isinstance(layer.post_attention_layernorm, Cosmos3NemotronRMSNorm)
        assert isinstance(transformer.norm, Cosmos3NemotronRMSNorm)
        assert not any("gate_proj" in key for key in state_dict)
        assert not any(".norm_q." in key or ".norm_k." in key for key in state_dict)
        assert "layers.0.self_attn.norm_added_q.weight" in state_dict
        assert "layers.0.self_attn.norm_added_k.weight" in state_dict
        assert "layers.0.self_attn.k_norm_und_for_gen.weight" in state_dict
        assert "layers.0.mlp.up_proj.weight" in state_dict
        assert "layers.0.mlp.down_proj.weight" in state_dict
        assert "action_proj_in.fc.weight" in state_dict
        assert "action_proj_out.fc.weight" in state_dict

    def test_cosmos3_edge_generator_k_norm_does_not_change_causal_attention(self):
        attention = Cosmos3PackedMoTAttention(
            hidden_size=12,
            head_dim=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_bias=False,
            rms_norm_eps=1e-5,
            qk_norm_for_text=False,
            use_und_k_norm_for_gen=True,
            norm_type="nemotron_rms_norm",
        ).to(torch_device)
        und_seq = torch.randn(3, 12, device=torch_device)
        gen_seq = torch.randn(2, 12, device=torch_device)
        rotary_emb = (
            torch.ones(3, 6, device=torch_device),
            torch.zeros(3, 6, device=torch_device),
            torch.ones(2, 6, device=torch_device),
            torch.zeros(2, 6, device=torch_device),
        )

        with torch.no_grad():
            causal_before, generation_before = attention(und_seq, gen_seq, rotary_emb)
            attention.k_norm_und_for_gen.weight.fill_(2)
            causal_after, generation_after = attention(und_seq, gen_seq, rotary_emb)

        torch.testing.assert_close(causal_before, causal_after)
        assert not torch.allclose(generation_before, generation_after)

    def test_cosmos3_edge_transformer_runs_action_workflow(self):
        transformer = self.model_class(
            **self.get_init_dict(), action_dim=3, action_gen=True, num_embodiment_domains=2
        ).eval()
        inputs = self.get_dummy_inputs()
        inputs["position_ids"] = torch.zeros((3, 4), dtype=torch.long, device=torch_device)
        inputs["sequence_length"] = 4
        inputs.update(
            {
                "action_tokens": [randn_tensor((1, 3), generator=self.generator, device=torch_device)],
                "action_token_shapes": [(1, 1, 1)],
                "action_sequence_indexes": torch.tensor([3], device=torch_device),
                "action_mse_loss_indexes": torch.tensor([3], device=torch_device),
                "action_timesteps": torch.tensor([1], device=torch_device),
                "action_noisy_frame_indexes": [torch.tensor([0], device=torch_device)],
                "action_domain_ids": [torch.tensor(0, device=torch_device)],
            }
        )

        with torch.no_grad():
            prediction, sound_prediction, action_prediction = transformer(**inputs, return_dict=False)

        assert prediction[0].shape == self.output_shape
        assert sound_prediction is None
        assert action_prediction[0].shape == (1, 3)

    def test_cosmos3_nemotron_rms_norm_multiplies_in_float32(self):
        hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        norm = Cosmos3NemotronRMSNorm(8, eps=1e-5).bfloat16()
        norm.weight.data.copy_(torch.randn(8, dtype=torch.bfloat16))

        expected = hidden_states.float()
        expected = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-5)
        expected = (norm.weight.float() * expected).to(hidden_states.dtype)

        torch.testing.assert_close(norm(hidden_states), expected, rtol=0, atol=0)


class TestCosmos3OmniTransformerMemory(Cosmos3OmniTransformerTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip("The transformer returns one tensor list per generated modality.")
    def test_layerwise_casting_training(self):
        super().test_layerwise_casting_training()


class TestCosmos3OmniTransformerTorchCompile(Cosmos3OmniTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        return super().get_dummy_inputs(height=height, width=width)


class TestCosmos3OmniTransformerTraining(Cosmos3OmniTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"Cosmos3OmniTransformer"})

    @pytest.mark.skip("The transformer returns one tensor list per generated modality.")
    def test_training(self):
        super().test_training()

    @pytest.mark.skip("The transformer returns one tensor list per generated modality.")
    def test_training_with_ema(self):
        super().test_training_with_ema()

    @pytest.mark.skip("The transformer returns one tensor list per generated modality.")
    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence()

    @pytest.mark.skip("The transformer returns one tensor list per generated modality.")
    def test_mixed_precision_training(self):
        super().test_mixed_precision_training()


class TestCosmos3OmniTransformerAttention(Cosmos3OmniTransformerTesterConfig, AttentionTesterMixin):
    pass
