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

import sys
import unittest

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GemmaForCausalLM

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Pipeline,
    Lumina2Transformer2DModel,
)

from ..testing_utils import floats_tensor, is_torch_version, require_peft_backend, skip_mps, torch_device


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class Lumina2LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = Lumina2Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "sample_size": 4,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 8,
        "num_layers": 2,
        "num_attention_heads": 1,
        "num_kv_heads": 1,
        "multiple_of": 16,
        "ffn_dim_multiplier": None,
        "norm_eps": 1e-5,
        "scaling_factor": 1.0,
        "axes_dim_rope": [4, 2, 2],
        "cap_feat_dim": 8,
    }
    transformer_cls = Lumina2Transformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    vae_cls = AutoencoderKL
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/dummy-gemma"
    text_encoder_cls, text_encoder_id = GemmaForCausalLM, "hf-internal-testing/dummy-gemma-diffusers"

    @property
    def output_shape(self):
        return (1, 4, 4, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in Lumina2.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Lumina2.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Lumina2.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Lumina2.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Lumina2.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Lumina2.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Lumina2.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Lumina2.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    @skip_mps
    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cpu" and is_torch_version(">=", "2.5"),
        reason="Test currently fails on CPU and PyTorch 2.5.1 but not on PyTorch 2.4.1.",
        strict=False,
    )
    def test_lora_fuse_nan(self):
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        # corrupt one LoRA weight with `inf` values
        with torch.no_grad():
            pipe.transformer.layers[0].attn.to_q.lora_A["adapter-1"].weight += float("inf")

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

        # without we should not see an error, but every image will be black
        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
        out = pipe(**inputs)[0]

        self.assertTrue(np.isnan(out).all())
