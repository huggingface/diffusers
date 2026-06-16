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
import sys
import unittest

import numpy as np
import torch
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Ideogram4Pipeline,
    Ideogram4Transformer2DModel,
)
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import QWEN3_VL_ACTIVATION_LAYERS

from ..testing_utils import floats_tensor, is_peft_available, require_peft_backend, torch_device


if is_peft_available():
    from peft import LoraConfig


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


# The text conditioning concatenates the hidden states of these Qwen3-VL decoder layers, so the dummy text
# encoder must be deep enough to expose the last tapped layer, and `llm_features_dim` must match the product.
_TEXT_HIDDEN_SIZE = 8
_NUM_TEXT_LAYERS = max(QWEN3_VL_ACTIVATION_LAYERS) + 1
_LLM_FEATURES_DIM = len(QWEN3_VL_ACTIVATION_LAYERS) * _TEXT_HIDDEN_SIZE


@require_peft_backend
class Ideogram4LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = Ideogram4Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "in_channels": 16,
        "num_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "adaln_dim": 16,
        "llm_features_dim": _LLM_FEATURES_DIM,
        "rope_theta": 10_000,
        "mrope_section": (2, 1, 1),
        "norm_eps": 1e-5,
    }
    transformer_cls = Ideogram4Transformer2DModel

    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",),
        "up_block_types": ("UpDecoderBlock2D",),
        "block_out_channels": (8,),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 1,
        "sample_size": 32,
        "patch_size": (2, 2),
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }
    vae_cls = AutoencoderKLFlux2

    tokenizer_cls, tokenizer_id = Qwen2Tokenizer, "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

    # Ideogram4's attention uses split q/k/v/out projections in the diffusers transformer.
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    # The text encoder (Qwen3-VL) is frozen and not LoRA-adapted by the Ideogram4 loader.
    supports_text_encoder_loras = False

    @property
    def output_shape(self):
        return (1, 16, 16, 3)

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        # The Ideogram4 pipeline takes a second (unconditional) transformer and a Qwen3-VL text encoder for
        # which there is no tiny pretrained checkpoint, so build the components inline rather than relying on
        # the base implementation.
        scheduler_cls = self.scheduler_cls if scheduler_cls is None else scheduler_cls
        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        torch.manual_seed(0)
        transformer = self.transformer_cls(**self.transformer_kwargs)
        unconditional_transformer = self.transformer_cls(**self.transformer_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)

        torch.manual_seed(0)
        text_config = {
            "hidden_size": _TEXT_HIDDEN_SIZE,
            "num_hidden_layers": _NUM_TEXT_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 256,
            "rope_theta": 10_000.0,
        }
        vision_config = {
            "hidden_size": 8,
            "depth": 2,
            "num_heads": 2,
            "intermediate_size": 16,
            "out_hidden_size": _TEXT_HIDDEN_SIZE,
            "patch_size": 14,
        }
        text_encoder = Qwen3VLModel(Qwen3VLConfig(text_config=text_config, vision_config=vision_config))
        tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_id)

        scheduler = scheduler_cls(**self.scheduler_kwargs)

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
            "unconditional_transformer": unconditional_transformer,
        }

        return pipeline_components, text_lora_config, denoiser_lora_config

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 32
        num_channels = 4
        sizes = (16, 16)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_schedule": [1.0, 1.0],
            "height": 16,
            "width": 16,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    # Overridden because the base test's rank-pattern module finder doesn't resolve a module on Ideogram4's
    # attention naming; this mirrors the same override other DiT LoRA tests use (e.g. Z-Image).
    def test_correct_lora_configs_with_different_ranks(self):
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")

        lora_output_same_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.transformer.delete_adapters("adapter-1")

        denoiser = pipe.unet if self.unet_kwargs is not None else pipe.transformer
        for name, _ in denoiser.named_modules():
            if "to_k" in name and "attention" in name and "lora" not in name:
                module_name_to_rank_update = name.replace(".base_layer.", ".")
                break

        # change the rank_pattern
        updated_rank = denoiser_lora_config.r * 2
        denoiser_lora_config.rank_pattern = {module_name_to_rank_update: updated_rank}

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        updated_rank_pattern = pipe.transformer.peft_config["adapter-1"].rank_pattern

        self.assertTrue(updated_rank_pattern == {module_name_to_rank_update: updated_rank})

        lora_output_diff_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(not np.allclose(original_output, lora_output_same_rank, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_rank, lora_output_same_rank, atol=1e-3, rtol=1e-3))

        pipe.transformer.delete_adapters("adapter-1")

        # similarly change the alpha_pattern
        updated_alpha = denoiser_lora_config.lora_alpha * 2
        denoiser_lora_config.alpha_pattern = {module_name_to_rank_update: updated_alpha}

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(
            pipe.transformer.peft_config["adapter-1"].alpha_pattern == {module_name_to_rank_update: updated_alpha}
        )

        lora_output_diff_alpha = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(not np.allclose(original_output, lora_output_diff_alpha, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_alpha, lora_output_same_rank, atol=1e-3, rtol=1e-3))

    @unittest.skip("Not supported in Ideogram4.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Ideogram4.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Ideogram4.")
    def test_modify_padding_mode(self):
        pass

    # Overridden because the base test probes for `transformer_blocks`/`blocks`/etc. to corrupt a weight,
    # but Ideogram4's transformer tower is named `layers` (with `attention.to_q` projections).
    def test_lora_fuse_nan(self):
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        # corrupt one LoRA weight with `inf` values
        with torch.no_grad():
            pipe.transformer.layers[0].attention.to_q.lora_A["adapter-1"].weight += float("inf")

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

        # without we should not see an error, but every image will be black
        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
        out = pipe(**inputs)[0]

        self.assertTrue(np.isnan(out).all())
