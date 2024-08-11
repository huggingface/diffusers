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

from transformers import AutoTokenizer, T5EncoderModel

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
class AuraFlowLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = AuraFlowPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    uses_flow_matching = True
    transformer_kwargs = {
        "sample_size": 64,
        "patch_size": 1,
        "in_channels": 4,
        "num_mmdit_layers": 1,
        "num_single_dit_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "caption_projection_dim": 32,
        "out_channels": 4,
        "pos_embed_max_size": 32,
    }
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

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
        "shift_factor": None,
        "scaling_factor": 0.13025,
    }

    @property
    def output_shape(self):
        return (1, 64, 64, 3)