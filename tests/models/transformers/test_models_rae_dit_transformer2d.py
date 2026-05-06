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

from diffusers import RAEDiT2DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_rae_dit import (
    RAEDiTAttention,
    RAEDiTAttnProcessor,
    VisionRotaryEmbedding,
    _expand_conditioning_tokens,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin, TrainingTesterMixin


enable_full_determinism()


def _initialize_non_zero_stage2_head(model: RAEDiT2DModel):
    torch.manual_seed(0)

    for block in model.blocks:
        block.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
        block.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)

    model.final_layer.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.bias.data.normal_(mean=0.0, std=0.02)


class RAEDiT2DTesterConfig(BaseModelTesterConfig):
    model_class = RAEDiT2DModel
    main_input_name = "hidden_states"
    input_shape = (8, 4, 4)
    output_shape = (8, 4, 4)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "sample_size": 4,
            "patch_size": 1,
            "in_channels": 8,
            "hidden_size": (32, 64),
            "depth": (2, 1),
            "num_heads": (4, 4),
            "mlp_ratio": 2.0,
            "class_dropout_prob": 0.0,
            "num_classes": 10,
            "use_qknorm": True,
            "use_swiglu": True,
            "use_rope": True,
            "use_rmsnorm": True,
            "wo_shift": False,
            "use_pos_embed": True,
        }

    def get_dummy_inputs(self):
        batch_size = 2
        in_channels = 8
        sample_size = 4
        scheduler_num_train_steps = 1000
        num_class_labels = 10

        hidden_states = randn_tensor(
            (batch_size, in_channels, sample_size, sample_size), generator=self.generator, device=torch_device
        )
        timesteps = torch.randint(0, scheduler_num_train_steps, size=(batch_size,), generator=self.generator).to(
            torch_device
        )
        class_labels = torch.randint(0, num_class_labels, size=(batch_size,), generator=self.generator).to(
            torch_device
        )

        return {"hidden_states": hidden_states, "timestep": timesteps, "class_labels": class_labels}


class TestRAEDiT2DModel(RAEDiT2DTesterConfig, ModelTesterMixin):
    def test_attention_processor_matches_reference_sdpa(self):
        attn = RAEDiTAttention(32, num_heads=4, qk_norm=True, use_rmsnorm=True).to(torch_device).eval()
        hidden_states = randn_tensor((2, 5, 32), generator=self.generator, device=torch_device)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.num_heads, attn.head_dim)).transpose(1, 2)
        key = key.unflatten(-1, (attn.num_heads, attn.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (attn.num_heads, attn.head_dim)).transpose(1, 2)

        query = attn.q_norm(query)
        key = attn.k_norm(key)
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)

        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        expected = expected.transpose(1, 2).reshape(hidden_states.shape)
        expected = attn.to_out[0](expected)
        expected = attn.to_out[1](expected)

        actual = attn(hidden_states)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_vision_rope_preserves_dispatch_attention_layout(self):
        rope = VisionRotaryEmbedding(dim=4, pt_seq_len=2).to(torch_device)
        hidden_states = randn_tensor((2, 4, 4, 8), generator=self.generator, device=torch_device)

        actual = rope(hidden_states)
        expected = apply_rotary_emb(
            hidden_states.transpose(1, 2),
            (rope.freqs_cos, rope.freqs_sin),
            sequence_dim=2,
        ).transpose(1, 2)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_attention_processor_plumbing(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        processors = model.attn_processors

        assert len(processors) == model.num_blocks
        assert all(isinstance(processor, RAEDiTAttnProcessor) for processor in processors.values())

        new_processor = RAEDiTAttnProcessor()
        model.set_attn_processor(new_processor)

        assert all(processor is new_processor for processor in model.attn_processors.values())

    def test_fuse_unfuse_qkv_projections_preserves_attention_output(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        attn = model.blocks[0].attn
        num_patches = model.s_embedder.height * model.s_embedder.width
        hidden_states = randn_tensor(
            (2, num_patches, model.encoder_hidden_size), generator=self.generator, device=torch_device
        )

        output_before_fusion = attn(hidden_states, rope=model.enc_feat_rope)

        attn.fuse_projections()

        assert attn.fused_projections
        assert hasattr(attn, "to_qkv")

        output_after_fusion = attn(hidden_states, rope=model.enc_feat_rope)

        assert torch.allclose(output_before_fusion, output_after_fusion, atol=1e-6, rtol=1e-5)

        attn.unfuse_projections()

        assert not attn.fused_projections
        assert not hasattr(attn, "to_qkv")
        assert torch.allclose(output_before_fusion, attn(hidden_states, rope=model.enc_feat_rope), atol=1e-6, rtol=1e-5)

    def test_model_fuse_unfuse_qkv_projections_preserves_output(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        _initialize_non_zero_stage2_head(model)
        inputs_dict = self.get_dummy_inputs()

        with torch.no_grad():
            output_before_fusion = model(**inputs_dict).sample

            model.fuse_qkv_projections()
            output_after_fusion = model(**inputs_dict).sample
            fused_projections_enabled = all(block.attn.fused_projections for block in model.blocks)

            model.unfuse_qkv_projections()
            output_after_unfusion = model(**inputs_dict).sample

        assert fused_projections_enabled
        assert torch.allclose(output_before_fusion, output_after_fusion, atol=1e-6, rtol=1e-5)
        assert all(not block.attn.fused_projections for block in model.blocks)
        assert torch.allclose(output_before_fusion, output_after_unfusion, atol=1e-6, rtol=1e-5)

    def test_attention_cpu_bfloat16_smoke(self):
        attn = RAEDiTAttention(32, num_heads=4, qk_norm=True, use_rmsnorm=True).to(dtype=torch.bfloat16).eval()
        hidden_states = torch.randn(2, 5, 32, dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype

    def test_swiglu_feedforward_matches_previous_chunk_order(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        block = model.blocks[0]

        hidden_states = randn_tensor((2, 4, model.encoder_hidden_size), generator=self.generator, device=torch_device)
        projection = block.mlp.net[0].proj
        output_projection = block.mlp.net[2]

        unswapped_weight = torch.cat(projection.weight.data.chunk(2, dim=0)[::-1], dim=0)
        unswapped_bias = None
        if projection.bias is not None:
            unswapped_bias = torch.cat(projection.bias.data.chunk(2, dim=0)[::-1], dim=0)

        projected = torch.nn.functional.linear(hidden_states, unswapped_weight, unswapped_bias)
        first_half, second_half = projected.chunk(2, dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(first_half) * second_half,
            output_projection.weight,
            output_projection.bias,
        )

        actual = block.mlp(hidden_states)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_output_with_precomputed_conditioning_hidden_states(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device).eval()
        _initialize_non_zero_stage2_head(model)

        batch_size = inputs_dict[self.main_input_name].shape[0]
        num_patches = (init_dict["sample_size"] // init_dict["patch_size"]) ** 2
        conditioning_hidden_states = randn_tensor(
            (batch_size, num_patches, init_dict["hidden_size"][0]), generator=self.generator, device=torch_device
        )

        with torch.no_grad():
            output = model(**inputs_dict, conditioning_hidden_states=conditioning_hidden_states).sample

        assert output.shape == inputs_dict[self.main_input_name].shape

    def test_precomputed_conditioning_hidden_states_match_model_dtype_with_identity_projector(self):
        init_dict = self.get_init_dict()
        init_dict["hidden_size"] = (32, 32)
        inputs_dict = self.get_dummy_inputs()
        model_dtype = torch.float16 if torch_device.startswith("cuda") else torch.bfloat16
        model = self.model_class(**init_dict).to(device=torch_device, dtype=model_dtype).eval()

        batch_size = inputs_dict[self.main_input_name].shape[0]
        num_patches = (init_dict["sample_size"] // init_dict["patch_size"]) ** 2
        conditioning_hidden_states = randn_tensor(
            (batch_size, num_patches, init_dict["hidden_size"][0]), generator=self.generator, device=torch_device
        ).float()

        inputs_dict["hidden_states"] = inputs_dict["hidden_states"].to(dtype=model_dtype)

        with torch.no_grad():
            output = model(**inputs_dict, conditioning_hidden_states=conditioning_hidden_states).sample

        assert output.shape == inputs_dict[self.main_input_name].shape
        assert output.dtype == model_dtype

    def test_precomputed_conditioning_matches_internal_encoder_path(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device).eval()
        _initialize_non_zero_stage2_head(model)

        hidden_states = inputs_dict["hidden_states"]
        timesteps = inputs_dict["timestep"]
        class_labels = inputs_dict["class_labels"]

        with torch.no_grad():
            timestep_emb = model.t_embedder(timesteps.reshape(-1).to(torch_device))
            class_emb = model.y_embedder(class_labels.reshape(-1).to(torch_device), train=False)
            conditioning = torch.nn.functional.silu(timestep_emb + class_emb)

            conditioning_hidden_states = model.s_embedder(hidden_states)
            if model.use_pos_embed:
                conditioning_hidden_states = conditioning_hidden_states + model.pos_embed

            for block_idx in range(model.num_encoder_blocks):
                conditioning_hidden_states = model.blocks[block_idx](
                    conditioning_hidden_states,
                    conditioning,
                    feat_rope=model.enc_feat_rope,
                )

            conditioning_hidden_states = torch.nn.functional.silu(
                timestep_emb.unsqueeze(1) + conditioning_hidden_states
            )

            output_internal = model(**inputs_dict).sample
            output_precomputed = model(
                **inputs_dict,
                conditioning_hidden_states=conditioning_hidden_states,
            ).sample

        assert torch.allclose(output_internal, output_precomputed, atol=1e-5, rtol=1e-4)

    def test_expand_conditioning_tokens_preserves_2d_layout(self):
        hidden_states = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

        repeated = _expand_conditioning_tokens(hidden_states, target_length=16)

        expected = torch.tensor(
            [
                [
                    [1.0],
                    [1.0],
                    [2.0],
                    [2.0],
                    [1.0],
                    [1.0],
                    [2.0],
                    [2.0],
                    [3.0],
                    [3.0],
                    [4.0],
                    [4.0],
                    [3.0],
                    [3.0],
                    [4.0],
                    [4.0],
                ]
            ]
        )
        assert torch.equal(repeated, expected)

    def test_expand_conditioning_tokens_downsamples_preserving_2d_layout(self):
        hidden_states = torch.arange(1.0, 17.0).reshape(1, 16, 1)

        reduced = _expand_conditioning_tokens(hidden_states, target_length=4)

        expected = torch.tensor([[[3.5], [5.5], [11.5], [13.5]]])
        assert torch.equal(reduced, expected)

    def test_initialize_weights_preserves_unspecialized_linear_layers(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device)
        sentinel_weight = torch.full_like(model.s_projector.weight, 0.1234)
        sentinel_bias = torch.full_like(model.s_projector.bias, -0.4321)
        model.s_projector.weight.data.copy_(sentinel_weight)
        model.s_projector.bias.data.copy_(sentinel_bias)

        model.initialize_weights()

        assert torch.equal(model.s_projector.weight, sentinel_weight)
        assert torch.equal(model.s_projector.bias, sentinel_bias)

    def test_initialize_weights_reinitializes_owned_layers_without_flipping_swiglu(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        hidden_states = randn_tensor((2, 4, model.encoder_hidden_size), generator=self.generator, device=torch_device)

        mlp_output_before = model.blocks[0].mlp(hidden_states)
        timestep_weight_before = model.t_embedder.mlp[0].weight.detach().clone()

        model.initialize_weights()

        assert not torch.equal(model.t_embedder.mlp[0].weight, timestep_weight_before)
        assert torch.allclose(model.blocks[0].mlp(hidden_states), mlp_output_before, atol=1e-6, rtol=1e-5)

    def test_expand_conditioning_tokens_broadcasts_global_conditioning(self):
        hidden_states = torch.tensor([[[1.0, 2.0]]])

        repeated = _expand_conditioning_tokens(hidden_states, target_length=4)

        expected = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
        assert torch.equal(repeated, expected)

    def test_expand_conditioning_tokens_rejects_incompatible_multi_token_layouts(self):
        hidden_states = torch.randn(1, 2, 4)

        with pytest.raises(ValueError):
            _expand_conditioning_tokens(hidden_states, target_length=8)


class TestRAEDiT2DTraining(RAEDiT2DTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"RAEDiT2DModel"})

    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence(loss_tolerance=1e-4)
