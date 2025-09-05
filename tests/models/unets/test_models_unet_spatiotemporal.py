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

import copy
import unittest

import torch

from diffusers import UNetSpatioTemporalConditionModel
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available

from ...testing_utils import (
    enable_full_determinism,
    floats_tensor,
    skip_mps,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


logger = logging.get_logger(__name__)

enable_full_determinism()


@skip_mps
class UNetSpatioTemporalConditionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNetSpatioTemporalConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 2
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_frames, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 1, 32)).to(torch_device)

        return {
            "sample": noise,
            "timestep": time_step,
            "encoder_hidden_states": encoder_hidden_states,
            "added_time_ids": self._get_add_time_ids(),
        }

    @property
    def input_shape(self):
        return (2, 2, 4, 32, 32)

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
    def noise_aug_strength(self):
        return 0.02

    @property
    def addition_time_embed_dim(self):
        return 32

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
            "projection_class_embeddings_input_dim": self.addition_time_embed_dim * 3,
            "addition_time_embed_dim": self.addition_time_embed_dim,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def _get_add_time_ids(self, do_classifier_free_guidance=True):
        add_time_ids = [self.fps, self.motion_bucket_id, self.noise_aug_strength]

        passed_add_embed_dim = self.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.addition_time_embed_dim * 3

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], device=torch_device)
        add_time_ids = add_time_ids.repeat(1, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    @unittest.skip("Number of Norm Groups is not configurable")
    def test_forward_with_norm_groups(self):
        pass

    @unittest.skip("Deprecated functionality")
    def test_model_attention_slicing(self):
        pass

    @unittest.skip("Not supported")
    def test_model_with_use_linear_projection(self):
        pass

    @unittest.skip("Not supported")
    def test_model_with_simple_projection(self):
        pass

    @unittest.skip("Not supported")
    def test_model_with_class_embeddings_concat(self):
        pass

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

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "TransformerSpatioTemporalModel",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "UNetMidBlockSpatioTemporal",
        }
        num_attention_heads = (8, 16)
        super().test_gradient_checkpointing_is_applied(
            expected_set=expected_set, num_attention_heads=num_attention_heads
        )

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
