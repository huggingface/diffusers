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

from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.utils.testing_utils import is_peft_available, require_peft_backend, require_torch_gpu, torch_device


if is_peft_available():
    pass

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class SD3LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = StableDiffusion3Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}
    uses_flow_matching = True
    transformer_kwargs = {
        "sample_size": 32,
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "caption_projection_dim": 32,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 64,
        "out_channels": 4,
    }
    transformer_cls = SD3Transformer2DModel
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
    has_three_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_2_cls, tokenizer_2_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_3_cls, tokenizer_3_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder"
    text_encoder_2_cls, text_encoder_2_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder-2"
    text_encoder_3_cls, text_encoder_3_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    @require_torch_gpu
    def test_sd3_lora(self):
        """
        Test loading the loras that are saved with the diffusers and peft formats.
        Related PR: https://github.com/huggingface/diffusers/pull/8584
        """
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components[0])
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        lora_model_id = "hf-internal-testing/tiny-sd3-loras"

        lora_filename = "lora_diffusers_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.unload_lora_weights()

        lora_filename = "lora_peft_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
