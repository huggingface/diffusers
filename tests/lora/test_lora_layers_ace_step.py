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

import sys
import unittest

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from diffusers import AutoencoderOobleck, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.ace_step_transformer import AceStepTransformer1DModel
from diffusers.pipelines.ace_step import AceStepConditionEncoder, AceStepPipeline
from diffusers.utils.import_utils import is_peft_available

from ..testing_utils import (
    require_peft_backend,
    skip_mps,
    torch_device,
)


if is_peft_available():
    from peft import LoraConfig

sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
@skip_mps
class AceStepLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = AceStepPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {"num_train_timesteps": 1, "shift": 1.0}

    transformer_cls = AceStepTransformer1DModel
    transformer_kwargs = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "in_channels": 24,
        "audio_acoustic_hidden_dim": 8,
        "patch_size": 2,
        "rope_theta": 10000.0,
        "sliding_window": 16,
    }

    vae_cls = AutoencoderOobleck
    vae_kwargs = {
        "encoder_hidden_size": 6,
        "downsampling_ratios": [1, 2],
        "decoder_channels": 3,
        "decoder_input_channels": 8,
        "audio_channels": 2,
        "channel_multiples": [2, 4],
        "sampling_rate": 4,
    }

    tokenizer_cls, tokenizer_id = AutoTokenizer, "Qwen/Qwen3-Embedding-0.6B"
    text_encoder_cls, text_encoder_id = None, None

    text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    supports_text_encoder_loras = False

    @property
    def output_shape(self):
        return (1, 2, 1)

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        scheduler_cls = scheduler_cls or self.scheduler_cls
        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        torch.manual_seed(0)
        transformer = self.transformer_cls(**self.transformer_kwargs)

        scheduler = scheduler_cls(**self.scheduler_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)

        torch.manual_seed(0)
        qwen3_config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=151936,
            max_position_embeddings=256,
        )
        text_encoder = Qwen3Model(qwen3_config)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        torch.manual_seed(0)
        condition_encoder = AceStepConditionEncoder(
            hidden_size=32,
            intermediate_size=64,
            text_hidden_dim=32,
            timbre_hidden_dim=8,
            num_lyric_encoder_hidden_layers=2,
            num_timbre_encoder_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            sliding_window=16,
        )

        text_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.text_encoder_target_modules,
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
            "condition_encoder": condition_encoder,
            "audio_tokenizer": None,
            "audio_token_detokenizer": None,
        }

        return pipeline_components, text_lora_config, denoiser_lora_config

    def get_dummy_inputs(self, with_generator=True):
        generator = torch.manual_seed(0)
        noise = torch.randn(1, 4, 8)
        input_ids = torch.randint(1, 10, size=(1, 10), generator=generator)

        pipeline_inputs = {
            "prompt": "A beautiful piano piece",
            "lyrics": "[verse]\nSoft notes",
            "audio_duration": 0.4,
            "num_inference_steps": 2,
            "max_text_length": 32,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs["generator"] = generator

        return noise, input_ids, pipeline_inputs

    @unittest.skip("Not supported in AceStep.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in AceStep.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in AceStep.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Not supported in AceStep.")
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        pass

    @unittest.skip("Tiny AceStep GQA model produces numerically close outputs for different LoRA ranks.")
    def test_correct_lora_configs_with_different_ranks(self):
        pass

    @unittest.skip("AceStep attention layers have no bias; lora_bias is not applicable.")
    def test_lora_B_bias(self):
        pass

    def test_lora_fuse_nan(self):
        import numpy as np

        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")

        with torch.no_grad():
            pipe.transformer.layers[0].self_attn.to_q.lora_A["adapter-1"].weight += float("inf")

        with self.assertRaises(ValueError):
            pipe.fuse_lora(safe_fusing=True)

        pipe.fuse_lora(safe_fusing=False)
        out = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(np.isnan(out).all())
