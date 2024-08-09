# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AuraFlowPipeline,
)
from diffusers.utils.testing_utils import is_peft_available, require_peft_backend, require_torch_gpu, torch_device


if is_peft_available():
    pass

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class AFLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = AuraFlowPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    transformer_kwargs = {
        "sample_size": 64,
        "patch_size": 2,
        "in_channels": 4,
        "num_mmdit_layers": 4,
        "num_single_dit_layers": 32,
        "attention_head_dim": 256,
        "num_attention_heads": 12,
        "joint_attention_dim": 2048,
        "caption_projection_dim": 3072,
        "out_channels": 4,
        "pos_embed_max_size": 1024,
    }
    vae_kwargs = {
        "sample_size": 1024,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": [
    128,
    256,
    512,
    512
  ],
        "layers_per_block": 2,
        "latent_channels": 4,
        "norm_num_groups": 32,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "shift_factor": None,
        "scaling_factor": 0.13025,
    }
    has_three_text_encoders = False

    @require_torch_gpu
    def test_af_lora(self):
        """
        Test loading the loras that are saved with the diffusers and peft formats.
        Related PR: https://github.com/huggingface/diffusers/pull/8584
        """
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        lora_model_id = "Warlord-K/gorkem-auraflow-lora"

        lora_filename = "pytorch_lora_weights.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.unload_lora_weights()

        lora_filename = "lora_peft_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
