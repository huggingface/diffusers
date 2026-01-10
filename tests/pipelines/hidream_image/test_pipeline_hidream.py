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

import unittest

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    HiDreamImagePipeline,
    HiDreamImageTransformer2DModel,
)

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class HiDreamImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = HiDreamImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "prompt_embeds", "negative_prompt_embeds"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params
    test_xformers_attention = False
    test_layerwise_casting = True
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HiDreamImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_channels=[32, 16],
            text_emb_dim=64,
            num_routed_experts=4,
            num_activated_experts=2,
            axes_dims_rope=(4, 2, 2),
            max_resolution=(32, 32),
            llama_layers=(0, 1),
        ).eval()
        torch.manual_seed(0)
        vae = AutoencoderKL(scaling_factor=0.3611, shift_factor=0.1159)
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
            max_position_embeddings=128,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        text_encoder_4 = LlamaForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        text_encoder_4.generation_config.pad_token_id = 1
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer_4 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        scheduler = FlowMatchEulerDiscreteScheduler()

        components = {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer_3": tokenizer_3,
            "text_encoder_4": text_encoder_4,
            "tokenizer_4": tokenizer_4,
            "transformer": transformer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs)[0]
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (128, 128, 3))

        # fmt: off
        expected_slice = np.array([0.4507, 0.5256, 0.4205, 0.5791, 0.4848, 0.4831, 0.4443, 0.5107, 0.6586, 0.3163, 0.7318, 0.5933, 0.6252, 0.5512, 0.5357, 0.5983])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = np.concatenate([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(np.allclose(generated_slice, expected_slice, atol=1e-3))

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-4)
