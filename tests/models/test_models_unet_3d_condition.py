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

import os
import tempfile
import unittest

import numpy as np
import torch

from diffusers.models import ModelMixin, UNet3DConditionModel
from diffusers.models.attention_processor import AttnProcessor, LoRAAttnProcessor
from diffusers.utils import (
    floats_tensor,
    logging,
    skip_mps,
    torch_device,
)
from diffusers.utils.import_utils import is_xformers_available

from .test_modeling_common import ModelTesterMixin


logger = logging.get_logger(__name__)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def create_lora_layers(model, mock_weights: bool = True):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        has_cross_attention = name.endswith("attn2.processor") and not (
            name.startswith("transformer_in") or "temp_attentions" in name.split(".")
        )
        cross_attention_dim = model.config.cross_attention_dim if has_cross_attention else None
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        elif name.startswith("transformer_in"):
            # Note that the `8 * ...` comes from: https://github.com/huggingface/diffusers/blob/7139f0e874f10b2463caa8cbd585762a309d12d6/src/diffusers/models/unet_3d_condition.py#L148
            hidden_size = 8 * model.config.attention_head_dim

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        lora_attn_procs[name] = lora_attn_procs[name].to(model.device)

        if mock_weights:
            # add 1 to weights to mock trained weights
            with torch.no_grad():
                lora_attn_procs[name].to_q_lora.up.weight += 1
                lora_attn_procs[name].to_k_lora.up.weight += 1
                lora_attn_procs[name].to_v_lora.up.weight += 1
                lora_attn_procs[name].to_out_lora.up.weight += 1

    return lora_attn_procs


@skip_mps
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
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
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

    # Overriding to set `norm_num_groups` needs to be different for this model.
    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 32

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

    def test_lora_processors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        lora_attn_procs = create_lora_layers(model)

        # make sure we can set a list of attention processors
        model.set_attn_processor(lora_attn_procs)
        model.to(torch_device)

        # test that attn processors can be set to itself
        model.set_attn_processor(model.attn_processors)

        with torch.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample1 - sample2).abs().max() < 3e-3
        assert (sample3 - sample4).abs().max() < 3e-3

        # sample 2 and sample 3 should be different
        assert (sample2 - sample3).abs().max() > 3e-3

    def test_lora_save_load(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with torch.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.to(torch_device)
            new_model.load_attn_procs(tmpdirname)

        with torch.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample - new_sample).abs().max() < 1e-4

        # LoRA and no LoRA should NOT be the same
        assert (sample - old_sample).abs().max() > 1e-4

    def test_lora_save_load_safetensors(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with torch.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, safe_serialization=True)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.to(torch_device)
            new_model.load_attn_procs(tmpdirname)

        with torch.no_grad():
            new_sample = new_model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample - new_sample).abs().max() < 1e-4

        # LoRA and no LoRA should NOT be the same
        assert (sample - old_sample).abs().max() > 1e-4

    def test_lora_save_safetensors_load_torch(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        lora_attn_procs = create_lora_layers(model, mock_weights=False)
        model.set_attn_processor(lora_attn_procs)
        # Saving as torch, properly reloads with directly filename
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.to(torch_device)
            new_model.load_attn_procs(tmpdirname, weight_name="pytorch_lora_weights.bin")

    def test_lora_save_torch_force_load_safetensors_error(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        lora_attn_procs = create_lora_layers(model, mock_weights=False)
        model.set_attn_processor(lora_attn_procs)
        # Saving as torch, properly reloads with directly filename
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.to(torch_device)
            with self.assertRaises(IOError) as e:
                new_model.load_attn_procs(tmpdirname, use_safetensors=True)
            self.assertIn("Error no file named pytorch_lora_weights.safetensors", str(e.exception))

    def test_lora_on_off(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        lora_attn_procs = create_lora_layers(model)
        model.set_attn_processor(lora_attn_procs)

        with torch.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample

        model.set_attn_processor(AttnProcessor())

        with torch.no_grad():
            new_sample = model(**inputs_dict).sample

        assert (sample - new_sample).abs().max() < 1e-4
        assert (sample - old_sample).abs().max() < 3e-3

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
