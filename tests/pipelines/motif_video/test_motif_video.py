# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from transformers import (
    AutoTokenizer,
    T5Gemma2Encoder,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, MotifVideoPipeline
from diffusers.guiders import AdaptiveProjectedGuidance
from diffusers.models.transformers.transformer_motif_video import MotifVideoTransformer3DModel
from diffusers.utils.testing_utils import enable_full_determinism

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class MotifVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = MotifVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "guidance_scale"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        # Build a tiny T5Gemma2Encoder to match the pipeline's expected text_encoder type
        text_config = T5Gemma2TextConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=1104,
            max_position_embeddings=128,
            head_dim=16,
            num_key_value_heads=2,
            dropout_rate=0.0,
        )
        encoder_config = T5Gemma2EncoderConfig(text_config=text_config)
        text_encoder = T5Gemma2Encoder(encoder_config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = MotifVideoTransformer3DModel(
            in_channels=33,
            out_channels=16,
            num_attention_heads=2,
            attention_head_dim=12,
            num_layers=1,
            num_single_layers=1,
            mlp_ratio=4.0,
            patch_size=1,
            patch_size_t=1,
            qk_norm="rms_norm",
            text_embed_dim=32,
            rope_axes_dim=(4, 4, 4),
        )

        guider = AdaptiveProjectedGuidance()

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": None,
            "guider": guider,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A test video",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
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
        video = pipe(**inputs).frames
        generated_video = video[0]

        self.assertEqual(generated_video.shape, (9, 16, 16, 3))
