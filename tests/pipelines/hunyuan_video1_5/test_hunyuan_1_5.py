# Copyright 2025 The HuggingFace Team.
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
from transformers import CLIPTokenizer, T5Config, T5EncoderModel

from diffusers import (
    AutoencoderKLHunyuanVideo15,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
)
from diffusers.guiders import ClassifierFreeGuidance

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class HunyuanVideo15PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = HunyuanVideo15Pipeline
    params = frozenset(
        [
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_frames",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "prompt_embeds_2",
            "prompt_embeds_mask_2",
            "negative_prompt_embeds_2",
            "negative_prompt_embeds_mask_2",
            "output_type",
        ]
    )
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = frozenset()
    test_attention_slicing = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = False
    supports_dduf = False

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideo15Transformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=num_layers,
            num_refiner_layers=1,
            mlp_ratio=2.0,
            patch_size=1,
            patch_size_t=1,
            text_embed_dim=16,
            text_embed_2_dim=8,
            image_embed_dim=12,
            rope_axes_dim=(2, 4, 4),
            target_size=16,
            task_type="t2v",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo15(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(16, 16, 16, 16),
            layers_per_block=1,
            spatial_compression_ratio=4,
            temporal_compression_ratio=2,
            scaling_factor=0.476986,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        torch.manual_seed(0)
        main_text_config = T5Config(
            d_model=16,
            d_kv=4,
            d_ff=64,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            vocab_size=64,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            is_encoder_decoder=False,
            use_cache=False,
            tie_word_embeddings=False,
        )
        text_encoder = T5EncoderModel(main_text_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        secondary_text_config = T5Config(
            d_model=8,
            d_kv=4,
            d_ff=32,
            num_layers=2,
            num_heads=2,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            vocab_size=32,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            is_encoder_decoder=False,
            use_cache=False,
            tie_word_embeddings=False,
        )
        text_encoder_2 = T5EncoderModel(secondary_text_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        guider = ClassifierFreeGuidance(guidance_scale=1.0)

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "guider": guider,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        torch.manual_seed(seed)
        batch_size = 1
        seq_len = 4
        seq_len_2 = 3
        text_embed_dim = 16
        text_embed_2_dim = 8

        prompt_embeds = torch.randn((batch_size, seq_len, text_embed_dim), device=device)
        prompt_embeds_mask = torch.ones((batch_size, seq_len), device=device)
        prompt_embeds_2 = torch.randn((batch_size, seq_len_2, text_embed_2_dim), device=device)
        prompt_embeds_mask_2 = torch.ones((batch_size, seq_len_2), device=device)

        inputs = {
            "generator": generator,
            "num_inference_steps": 2,
            "num_frames": 5,
            "height": 16,
            "width": 16,
            "output_type": "pt",
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_embeds_mask_2": prompt_embeds_mask_2,
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        result = pipe(**inputs)
        video = result.frames

        generated_video = video[0]
        self.assertEqual(generated_video.shape, (inputs["num_frames"], 3, inputs["height"], inputs["width"]))
        self.assertFalse(torch.isnan(generated_video).any())
