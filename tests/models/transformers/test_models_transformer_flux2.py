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

from diffusers import Flux2Transformer2DModel, attention_backend
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


class Flux2TransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = Flux2Transformer2DModel
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
        sequence_length = 48
        embedding_dim = 32

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)  # [height * width, 4]
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            # "pooled_projections": pooled_prompt_embeds,
            "timestep": timestep,
            "guidance": guidance,
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
            "timestep_guidance_channels": 256,  # Hardcoded in original code
            "axes_dims_rope": [4, 4, 4, 4],
        }

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_flux2_consistency(self, seed=0):
        torch.manual_seed(seed)
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(seed)
        model = self.model_class(**init_dict)
        # state_dict = model.state_dict()
        # for key, param in state_dict.items():
        #     print(f"{key} | {param.shape}")
        # torch.save(state_dict, "/raid/daniel_gu/test_flux2_params/diffusers.pt")
        model.to(torch_device)
        model.eval()

        with attention_backend("native"):
            with torch.no_grad():
                output = model(**inputs_dict)

                if isinstance(output, dict):
                    output = output.to_tuple()[0]

        self.assertIsNotNone(output)

        # input & output have to have the same shape
        input_tensor = inputs_dict[self.main_input_name]
        expected_shape = input_tensor.shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

        # Check against expected slice
        # fmt: off
        expected_slice = torch.tensor([-0.3662, 0.4844, 0.6334, -0.3497, 0.2162, 0.0188, 0.0521, -0.2061, -0.2041, -0.0342, -0.7107, 0.4797, -0.3280, 0.7059, -0.0849, 0.4416])
        # fmt: on

        flat_output = output.cpu().flatten()
        generated_slice = torch.cat([flat_output[:8], flat_output[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-4))

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Flux2Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # The test exists for cases like
    # https://github.com/huggingface/diffusers/issues/11874
    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_lora_exclude_modules(self):
        from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict

        lora_rank = 4
        target_module = "single_transformer_blocks.0.attn.to_out"
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
            r=lora_rank, target_modules=[target_module], exclude_modules=["to_out"]
        )
        inject_adapter_in_model(config, model, adapter_name=adapter_name, state_dict=lora_state_dict)
        set_peft_model_state_dict(model, lora_state_dict, adapter_name)
        retrieved_lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
        assert len(retrieved_lora_state_dict) == len(lora_state_dict)
        assert (retrieved_lora_state_dict[f"{target_module}.lora_A.weight"] == 22).all()
        assert (retrieved_lora_state_dict[f"{target_module}.lora_B.weight"] == 33).all()


class Flux2TransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = Flux2Transformer2DModel
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def prepare_init_args_and_inputs_for_common(self):
        return Flux2TransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return Flux2TransformerTests().prepare_dummy_input(height=height, width=width)


class Flux2TransformerLoRAHotSwapTests(LoraHotSwappingForModelTesterMixin, unittest.TestCase):
    model_class = Flux2Transformer2DModel
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def prepare_init_args_and_inputs_for_common(self):
        return Flux2TransformerTests().prepare_init_args_and_inputs_for_common()

    def prepare_dummy_input(self, height, width):
        return Flux2TransformerTests().prepare_dummy_input(height=height, width=width)
