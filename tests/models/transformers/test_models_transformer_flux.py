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

import unittest

import torch

from diffusers import FluxTransformer2DModel
from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
from diffusers.models.embeddings import ImageProjection

from ...testing_utils import enable_full_determinism, is_peft_available, torch_device
from ..test_modeling_common import LoraHotSwappingForModelTesterMixin, ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


def create_flux_ip_adapter_state_dict(model):
    # "ip_adapter" (cross-attention weights)
    ip_cross_attn_state_dict = {}
    key_id = 0

    for name in model.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            continue

        joint_attention_dim = model.config["joint_attention_dim"]
        hidden_size = model.config["num_attention_heads"] * model.config["attention_head_dim"]
        sd = FluxIPAdapterJointAttnProcessor2_0(
            hidden_size=hidden_size, cross_attention_dim=joint_attention_dim, scale=1.0
        ).state_dict()
        ip_cross_attn_state_dict.update(
            {
                f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                f"{key_id}.to_k_ip.bias": sd["to_k_ip.0.bias"],
                f"{key_id}.to_v_ip.bias": sd["to_v_ip.0.bias"],
            }
        )

        key_id += 1

    # "image_proj" (ImageProjection layer weights)

    image_projection = ImageProjection(
        cross_attention_dim=model.config["joint_attention_dim"],
        image_embed_dim=(
            model.config["pooled_projection_dim"] if "pooled_projection_dim" in model.config.keys() else 768
        ),
        num_image_text_embeds=4,
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


class FluxTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = FluxTransformer2DModel
    main_input_name = "hidden_states"
    # We override the items here because the transformer under consideration is small.
    model_split_percents = [0.7, 0.6, 0.6]

    # Skip setting testing with default: AttnProcessor
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        return self.prepare_dummy_input()

    @property
    def input_shape(self):
        return (16, 4)

    @property
    def output_shape(self):
        return (16, 4)

    def prepare_dummy_input(self, height=4, width=4):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        sequence_length = 48
        embedding_dim = 32

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(torch_device)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(torch_device)
        image_ids = torch.randn((height * width, num_image_channels)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
        }

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 32,
            "axes_dims_rope": [4, 4, 8],
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_deprecated_inputs_img_txt_ids_3d(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output_1 = model(**inputs_dict).to_tuple()[0]

        # update inputs_dict with txt_ids and img_ids as 3d tensors (deprecated)
        text_ids_3d = inputs_dict["txt_ids"].unsqueeze(0)
        image_ids_3d = inputs_dict["img_ids"].unsqueeze(0)

        assert text_ids_3d.ndim == 3, "text_ids_3d should be a 3d tensor"
        assert image_ids_3d.ndim == 3, "img_ids_3d should be a 3d tensor"

        inputs_dict["txt_ids"] = text_ids_3d
        inputs_dict["img_ids"] = image_ids_3d

        with torch.no_grad():
            output_2 = model(**inputs_dict).to_tuple()[0]

        self.assertEqual(output_1.shape, output_2.shape)
        self.assertTrue(
            torch.allclose(output_1, output_2, atol=1e-5),
            msg="output with deprecated inputs (img_ids and txt_ids as 3d torch tensors) are not equal as them as 2d inputs",
        )

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"FluxTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # The test exists for cases like
    # https://github.com/huggingface/diffusers/issues/11874
    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_lora_exclude_modules(self):
        from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict

        lora_rank = 4
        target_module = "single_transformer_blocks.0.proj_out"
        adapter_name = "foo"
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        state_dict = model.state_dict()
        target_mod_shape = state_dict[f"{target_module}.weight"].shape
        lora_state_dict = {
            f"{target_module}.lora_A.weight": torch.ones(lora_rank, target_mod_shape[1]) * 22,
            f"{target_module}.lora_B.weight": torch.ones(target_mod_shape[0], lora_rank) * 33,
        }
        # Passing exclude_modules should no longer be necessary (or even passing target_modules, for that matter).
        config = LoraConfig(
            r=lora_rank, target_modules=["single_transformer_blocks.0.proj_out"], exclude_modules=["proj_out"]
        )
        inject_adapter_in_model(config, model, adapter_name=adapter_name, state_dict=lora_state_dict)
        set_peft_model_state_dict(model, lora_state_dict, adapter_name)
        retrieved_lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
        assert len(retrieved_lora_state_dict) == len(lora_state_dict)
        assert (retrieved_lora_state_dict["single_transformer_blocks.0.proj_out.lora_A.weight"] == 22).all()
        assert (retrieved_lora_state_dict["single_transformer_blocks.0.proj_out.lora_B.weight"] == 33).all()


class FluxTransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = FluxTransformer2DModel
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def prepare_init_args_and_inputs_for_common(self):
        return FluxTransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return FluxTransformerTests().prepare_dummy_input(height=height, width=width)


class FluxTransformerLoRAHotSwapTests(LoraHotSwappingForModelTesterMixin, unittest.TestCase):
    model_class = FluxTransformer2DModel
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def prepare_init_args_and_inputs_for_common(self):
        return FluxTransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return FluxTransformerTests().prepare_dummy_input(height=height, width=width)
