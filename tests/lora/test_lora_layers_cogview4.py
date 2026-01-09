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
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized
from transformers import AutoTokenizer, GlmModel

from diffusers import AutoencoderKL, CogView4Pipeline, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler

from ..testing_utils import (
    floats_tensor,
    require_peft_backend,
    require_torch_accelerator,
    skip_mps,
    torch_device,
)


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


class TokenizerWrapper:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-cogview4", subfolder="tokenizer", trust_remote_code=True
        )


@require_peft_backend
@skip_mps
class CogView4LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = CogView4Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": 2,
        "in_channels": 4,
        "num_layers": 2,
        "attention_head_dim": 4,
        "num_attention_heads": 4,
        "out_channels": 4,
        "text_embed_dim": 32,
        "time_embed_dim": 8,
        "condition_dim": 4,
    }
    transformer_cls = CogView4Transformer2DModel
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
        "sample_size": 128,
    }
    vae_cls = AutoencoderKL
    tokenizer_cls, tokenizer_id, tokenizer_subfolder = (
        TokenizerWrapper,
        "hf-internal-testing/tiny-random-cogview4",
        "tokenizer",
    )
    text_encoder_cls, text_encoder_id, text_encoder_subfolder = (
        GlmModel,
        "hf-internal-testing/tiny-random-cogview4",
        "text_encoder",
    )

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        sizes = (4, 4)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "",
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    def test_simple_inference_save_pretrained(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained
        """
        components, _, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)

            pipe_from_pretrained = self.pipeline_class.from_pretrained(tmpdirname)
            pipe_from_pretrained.to(torch_device)

        images_lora_save_pretrained = pipe_from_pretrained(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(images_lora, images_lora_save_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    @parameterized.expand([("block_level", True), ("leaf_level", False)])
    @require_torch_accelerator
    def test_group_offloading_inference_denoiser(self, offload_type, use_stream):
        # TODO: We don't run the (leaf_level, True) test here that is enabled for other models.
        # The reason for this can be found here: https://github.com/huggingface/diffusers/pull/11804#issuecomment-3013325338
        super()._test_group_offloading_inference_denoiser(offload_type, use_stream)

    @unittest.skip("Not supported in CogView4.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in CogView4.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in CogView4.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogView4.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogView4.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogView4.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogView4.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in CogView4.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass
