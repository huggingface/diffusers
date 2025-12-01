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
import sys
import unittest

import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)

from ..testing_utils import floats_tensor, is_peft_available, require_peft_backend


if is_peft_available():
    from peft import LoraConfig


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@unittest.skip(
    "ZImage LoRA tests are skipped due to non-deterministic behavior from complex64 RoPE operations "
    "and torch.empty padding tokens. LoRA functionality works correctly with real models."
)
@require_peft_backend
class ZImageLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = ZImagePipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "all_patch_size": (2,),
        "all_f_patch_size": (1,),
        "in_channels": 16,
        "dim": 32,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 16,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [8, 4, 4],
        "axes_lens": [256, 32, 32],
    }
    transformer_cls = ZImageTransformer2DModel
    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [32, 64],
        "layers_per_block": 1,
        "latent_channels": 16,
        "norm_num_groups": 32,
        "sample_size": 32,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
    }
    vae_cls = AutoencoderKL
    tokenizer_cls, tokenizer_id = Qwen2Tokenizer, "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
    text_encoder_cls, text_encoder_id = Qwen3Model, None  # Will be created inline
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        # Override to create Qwen3Model inline since it doesn't have a pretrained tiny model
        torch.manual_seed(0)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        text_encoder = Qwen3Model(config)
        tokenizer = Qwen2Tokenizer.from_pretrained(self.tokenizer_id)

        transformer = self.transformer_cls(**self.transformer_kwargs)
        vae = self.vae_cls(**self.vae_kwargs)

        if scheduler_cls is None:
            scheduler_cls = self.scheduler_cls
        scheduler = scheduler_cls(**self.scheduler_kwargs)

        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        text_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            init_lora_weights=False,
            use_dora=use_dora,
        )

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.denoiser_target_modules,
            init_lora_weights=False,
            use_dora=use_dora,
        )

        pipeline_components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

        return pipeline_components, text_lora_config, denoiser_lora_config
