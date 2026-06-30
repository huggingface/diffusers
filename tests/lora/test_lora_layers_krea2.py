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
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)

from ..testing_utils import floats_tensor, is_peft_available, require_peft_backend


if is_peft_available():
    from peft import LoraConfig


from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class Krea2LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = Krea2Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 6400,
    }

    transformer_cls = Krea2Transformer2DModel
    transformer_kwargs = {
        "in_channels": 16,
        "num_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "timestep_embed_dim": 8,
        "text_hidden_dim": 16,
        "num_text_layers": 3,
        "text_num_attention_heads": 2,
        "text_num_key_value_heads": 1,
        "text_intermediate_size": 16,
        "num_layerwise_text_blocks": 1,
        "num_refiner_text_blocks": 1,
        "axes_dims_rope": (4, 2, 2),
        "rope_theta": 1000.0,
    }

    z_dim = 4
    vae_cls = AutoencoderKLQwenImage
    vae_kwargs = {
        "base_dim": z_dim * 6,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }

    tokenizer_cls, tokenizer_id = Qwen2Tokenizer, "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

    # Krea2's attention uses split q/k/v/out projections in the diffusers transformer.
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    # The text encoder (Qwen3-VL) is frozen and not LoRA-adapted by the Krea2 loader.
    supports_text_encoder_loras = False

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        # The Krea2 pipeline uses a Qwen3-VL text encoder for which there is no tiny pretrained checkpoint,
        # so build the components inline rather than relying on the base implementation.
        scheduler_cls = self.scheduler_cls if scheduler_cls is None else scheduler_cls
        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        torch.manual_seed(0)
        transformer = self.transformer_cls(**self.transformer_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)

        torch.manual_seed(0)
        scheduler = scheduler_cls(**self.scheduler_kwargs)

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            vocab_size=152064,
        )
        text_encoder = Qwen3VLModel(config).eval()
        tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_id)

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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "text_encoder_select_layers": (0, 1, 2),
        }

        return pipeline_components, text_lora_config, denoiser_lora_config

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in Krea2.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Krea2.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Krea2.")
    def test_modify_padding_mode(self):
        pass
