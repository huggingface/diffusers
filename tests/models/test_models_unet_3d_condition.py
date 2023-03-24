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

import unittest

import numpy as np
import torch

from diffusers.models import ModelMixin, UNet3DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils import (
    floats_tensor,
    logging,
    torch_device,
)
from diffusers.utils.import_utils import is_xformers_available

from ..test_modeling_common import ModelTesterMixin


logger = logging.get_logger(__name__)
torch.backends.cuda.matmul.allow_tf32 = False


def create_lora_layers(model):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        lora_attn_procs[name] = lora_attn_procs[name].to(model.device)

        # add 1 to weights to mock trained weights
        with torch.no_grad():
            lora_attn_procs[name].to_q_lora.up.weight += 1
            lora_attn_procs[name].to_k_lora.up.weight += 1
            lora_attn_procs[name].to_v_lora.up.weight += 1
            lora_attn_procs[name].to_out_lora.up.weight += 1

    return lora_attn_procs


class UNet3DConditionModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet3DConditionModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 4, 32, 32)

    @property
    def output_shape(self):
        return (4, 4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64, 64, 64),
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 4,
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

    # Overriding because `block_out_channels` needs to be different for this model.
    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 32
        init_dict["block_out_channels"] = (32, 64, 64, 64)

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

    # Overriding since the UNet3D outputs a different structure.
    def test_determinism(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            # Warmup pass when using mps (see #372)
            if torch_device == "mps" and isinstance(model, ModelMixin):
                model(**self.dummy_input)

            first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first.sample

            second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second.sample

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, 1e-5)

    def test_model_attention_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

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

    # (`attn_processors`) needs to be implemented in this model for this test.
    # def test_lora_processors(self):

    # (`attn_processors`) needs to be implemented in this model for this test.
    # def test_lora_save_load(self):

    # (`attn_processors`) needs to be implemented for this test in the model.
    # def test_lora_save_load_safetensors(self):

    # (`attn_processors`) needs to be implemented for this test in the model.
    # def test_lora_save_safetensors_load_torch(self):

    # (`attn_processors`) needs to be implemented for this test.
    # def test_lora_save_torch_force_load_safetensors_error(self):

    # (`attn_processors`) needs to be added for this test.
    # def test_lora_on_off(self):

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_lora_xformers_on_off(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 4

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)
        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)

        # default
        with torch.no_grad():
            sample = model(**inputs_dict).sample

            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample

            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample

        assert (sample - on_sample).abs().max() < 1e-4
        assert (sample - off_sample).abs().max() < 1e-4


# (todo: sayakpaul) implement SLOW tests.
