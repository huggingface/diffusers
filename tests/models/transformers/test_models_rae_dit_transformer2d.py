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

import unittest

import torch

from diffusers import RAEDiTTransformer2DModel

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


def _initialize_non_zero_stage2_head(model: RAEDiTTransformer2DModel):
    torch.manual_seed(0)

    for block in model.blocks:
        block.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
        block.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)

    model.final_layer.adaLN_modulation[-1].weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.adaLN_modulation[-1].bias.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.weight.data.normal_(mean=0.0, std=0.02)
    model.final_layer.linear.bias.data.normal_(mean=0.0, std=0.02)


class RAEDiTTransformer2DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = RAEDiTTransformer2DModel
    main_input_name = "hidden_states"

    @property
    def dummy_input(self):
        batch_size = 2
        in_channels = 8
        sample_size = 4
        scheduler_num_train_steps = 1000
        num_class_labels = 10

        hidden_states = floats_tensor((batch_size, in_channels, sample_size, sample_size)).to(torch_device)
        timesteps = torch.randint(0, scheduler_num_train_steps, size=(batch_size,)).to(torch_device)
        class_labels = torch.randint(0, num_class_labels, size=(batch_size,)).to(torch_device)

        return {"hidden_states": hidden_states, "timestep": timesteps, "class_labels": class_labels}

    @property
    def input_shape(self):
        return (8, 4, 4)

    @property
    def output_shape(self):
        return (8, 4, 4)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
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
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        super().test_output(
            expected_output_shape=(self.dummy_input[self.main_input_name].shape[0],) + self.output_shape
        )

    def test_output_with_precomputed_conditioning_hidden_states(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
        _initialize_non_zero_stage2_head(model)

        batch_size = inputs_dict[self.main_input_name].shape[0]
        num_patches = (init_dict["sample_size"] // init_dict["patch_size"]) ** 2
        conditioning_hidden_states = floats_tensor((batch_size, num_patches, init_dict["hidden_size"][0])).to(
            torch_device
        )

        with torch.no_grad():
            output = model(**inputs_dict, conditioning_hidden_states=conditioning_hidden_states).sample

        self.assertEqual(output.shape, inputs_dict[self.main_input_name].shape)

    def test_precomputed_conditioning_matches_internal_encoder_path(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()
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

        self.assertTrue(torch.allclose(output_internal, output_precomputed, atol=1e-5, rtol=1e-4))

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"RAEDiTTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_effective_gradient_checkpointing(self):
        super().test_effective_gradient_checkpointing(loss_tolerance=1e-4)

    @unittest.skip("RAEDiT initializes the output head to zeros, so cosine-based layerwise casting checks are uninformative.")
    def test_layerwise_casting_inference(self):
        pass
