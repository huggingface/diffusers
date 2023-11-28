# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import copy
import unittest

import torch

from diffusers import UNetSpatioTemporalConditionModel
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_all_close,
    torch_device,
)

from .test_modeling_common import ModelTesterMixin, UNetTesterMixin


logger = logging.get_logger(__name__)

enable_full_determinism()


class UNetSpatioTemporalConditionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNetSpatioTemporalConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32)).to(torch_device)

        return {
            "sample": noise,
            "timestep": time_step,
            "encoder_hidden_states": encoder_hidden_states,
        }

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    @property
    def fps(self):
        return 6

    @property
    def motion_bucket_id(self):
        return 127

    @property
    def cond_aug(self):
        return 0.02

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            "up_block_types": (
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            "cross_attention_dim": 32,
            "num_attention_heads": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_enable_works(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        model.enable_xformers_memory_efficient_attention()

        assert (
            model.mid_block.attentions[0].transformer_blocks[0].attn1.processor.__class__.__name__
            == "XFormersAttnProcessor"
        ), "xformers is not enabled"

    @unittest.skipIf(torch_device == "mps", "Gradient checkpointing skipped on MPS")
    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()

        labels = torch.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        for name, param in named_params.items():
            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=5e-5))

    def test_model_with_num_attention_heads_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["num_attention_heads"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_use_linear_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["use_linear_projection"] = True

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_cross_attention_dim_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["cross_attention_dim"] = (32, 32)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_simple_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        batch_size, _, _, sample_size = inputs_dict["sample"].shape

        init_dict["class_embed_type"] = "simple_projection"
        init_dict["projection_class_embeddings_input_dim"] = sample_size

        inputs_dict["class_labels"] = floats_tensor((batch_size, sample_size)).to(torch_device)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_class_embeddings_concat(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        batch_size, _, _, sample_size = inputs_dict["sample"].shape

        init_dict["class_embed_type"] = "simple_projection"
        init_dict["projection_class_embeddings_input_dim"] = sample_size
        init_dict["class_embeddings_concat"] = True

        inputs_dict["class_labels"] = floats_tensor((batch_size, sample_size)).to(torch_device)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_attention_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["num_attention_heads"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model.set_attention_slice("auto")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice("max")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice(2)
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

    def test_model_sliceable_head_dim(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["num_attention_heads"] = (8, 16)

        model = self.model_class(**init_dict)

        def check_sliceable_dim_attr(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                assert isinstance(module.sliceable_head_dim, int)

            for child in module.children():
                check_sliceable_dim_attr(child)

        # retrieve number of attention layers
        for module in model.children():
            check_sliceable_dim_attr(module)

    def test_gradient_checkpointing_is_applied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["num_attention_heads"] = (8, 16)

        model_class_copy = copy.copy(self.model_class)

        modules_with_gc_enabled = {}

        # now monkey patch the following function:
        #     def _set_gradient_checkpointing(self, module, value=False):
        #         if hasattr(module, "gradient_checkpointing"):
        #             module.gradient_checkpointing = value

        def _set_gradient_checkpointing_new(self, module, value=False):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                modules_with_gc_enabled[module.__class__.__name__] = True

        model_class_copy._set_gradient_checkpointing = _set_gradient_checkpointing_new

        model = model_class_copy(**init_dict)
        model.enable_gradient_checkpointing()

        EXPECTED_SET = {
            "CrossAttnUpBlock2D",
            "CrossAttnDownBlock2D",
            "UNetMidBlock2DCrossAttn",
            "UpBlock2D",
            "Transformer2DModel",
            "DownBlock2D",
        }

        assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    def test_pickle(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["num_attention_heads"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample = model(**inputs_dict).sample

        sample_copy = copy.copy(sample)

        assert (sample - sample_copy).abs().max() < 1e-4
