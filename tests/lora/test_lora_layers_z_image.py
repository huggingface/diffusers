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
import os
import sys
import unittest

import numpy as np
import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)

from ..testing_utils import floats_tensor, is_peft_available, require_peft_backend, torch_device


if is_peft_available():
    from peft import LoraConfig


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


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

    @unittest.skip("Not supported in ZImage.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in ZImage.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in ZImage.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ZImage.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ZImage.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ZImage.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ZImage.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ZImage.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    @unittest.skip("Not supported in ZImage.")
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        pass

    def test_lora_fuse_nan(self):
        """Override to use ZImage's 'layers' attribute instead of 'transformer_blocks'."""
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        denoiser = pipe.transformer
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        # corrupt one LoRA weight with `inf` values - ZImage uses 'layers.X.attention'
        with torch.no_grad():
            pipe.transformer.layers[0].attention.to_q.lora_A["adapter-1"].weight += float("inf")

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

        # without we should not see an error, but every image will be black
        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
        out = pipe(**inputs)[0]

        self.assertTrue(np.isnan(out).all())

    def test_correct_lora_configs_with_different_ranks(self):
        """Override to use ZImage's 'attention' naming instead of 'attn'."""
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_output_same_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]
        pipe.transformer.delete_adapters("adapter-1")

        # ZImage uses 'attention.to_k' not 'attn.to_k'
        denoiser = pipe.transformer
        for name, _ in denoiser.named_modules():
            if "to_k" in name and "attention" in name and "lora" not in name:
                module_name_to_rank_update = name.replace(".base_layer.", ".")
                break

        updated_rank = denoiser_lora_config.r * 2
        denoiser_lora_config.rank_pattern = {module_name_to_rank_update: updated_rank}

        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        lora_output_different_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(
            np.allclose(original_output, lora_output_same_rank, atol=1e-3, rtol=1e-3),
            "LoRA should change the output.",
        )
        self.assertFalse(
            np.allclose(lora_output_same_rank, lora_output_different_rank, atol=1e-3, rtol=1e-3),
            "Different LoRA ranks should produce different outputs.",
        )
