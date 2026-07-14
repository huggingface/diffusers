# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers import Cosmos3OmniTransformer
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm


def get_edge_transformer(action_gen: bool = False):
    return Cosmos3OmniTransformer(
        action_dim=3 if action_gen else None,
        action_gen=action_gen,
        backbone_type="cosmos3_edge_nemotron_dense",
        head_dim=4,
        hidden_act="relu2",
        hidden_size=16,
        intermediate_size=32,
        latent_channel=1,
        latent_patch_size=1,
        num_attention_heads=4,
        num_embodiment_domains=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        patch_latent_dim=1,
        qk_norm_for_text=False,
        rms_norm_eps=1e-5,
        rope_theta=1e8,
        temporal_compression_factor=4,
        vocab_size=32,
    )


def test_cosmos3_edge_uses_nemotron_parameter_layout():
    transformer = get_edge_transformer(action_gen=True)
    state_dict = transformer.state_dict()
    layer = transformer.layers[0]

    assert transformer.config.backbone_type == "cosmos3_edge_nemotron_dense"
    assert transformer.config.temporal_compression_factor == 4
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
    assert "layers.0.mlp.up_proj.weight" in state_dict
    assert "layers.0.mlp.down_proj.weight" in state_dict
    assert "action_proj_in.fc.weight" in state_dict
    assert "action_proj_out.fc.weight" in state_dict


def test_cosmos3_edge_transformer_runs_action_workflow():
    transformer = get_edge_transformer(action_gen=True).eval()

    with torch.no_grad():
        prediction, sound_prediction, action_prediction = transformer(
            input_ids=torch.tensor([1, 2]),
            text_indexes=torch.tensor([0, 1]),
            position_ids=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]),
            und_len=2,
            sequence_length=4,
            vision_tokens=[torch.randn(1, 1, 1, 1, 1)],
            vision_token_shapes=[(1, 1, 1)],
            vision_sequence_indexes=torch.tensor([2]),
            vision_mse_loss_indexes=torch.tensor([2]),
            vision_timesteps=torch.tensor([1]),
            vision_noisy_frame_indexes=[torch.tensor([0])],
            action_tokens=[torch.randn(1, 3)],
            action_token_shapes=[(1, 1, 1)],
            action_sequence_indexes=torch.tensor([3]),
            action_mse_loss_indexes=torch.tensor([3]),
            action_timesteps=torch.tensor([1]),
            action_noisy_frame_indexes=[torch.tensor([0])],
            action_domain_ids=[torch.tensor(0)],
        )

    assert prediction[0].shape == (1, 1, 1, 1, 1)
    assert sound_prediction is None
    assert action_prediction[0].shape == (1, 3)


def test_cosmos3_nemotron_rms_norm_multiplies_in_float32():
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    norm = Cosmos3NemotronRMSNorm(8, eps=1e-5).bfloat16()
    norm.weight.data.copy_(torch.randn(8, dtype=torch.bfloat16))

    expected = hidden_states.float()
    expected = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = (norm.weight.float() * expected).to(hidden_states.dtype)

    torch.testing.assert_close(norm(hidden_states), expected, rtol=0, atol=0)
