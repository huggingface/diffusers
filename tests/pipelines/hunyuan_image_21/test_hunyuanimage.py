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

import numpy as np
import torch
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AdaptiveProjectedMixGuidance,
    AutoencoderKLHunyuanImage,
    FlowMatchEulerDiscreteScheduler,
    HunyuanImagePipeline,
    HunyuanImageTransformer2DModel,
)

from ...testing_utils import enable_full_determinism
from ..test_pipelines_common import (
    FirstBlockCacheTesterMixin,
    PipelineTesterMixin,
    to_np,
)


enable_full_determinism()


class HunyuanImagePipelineFastTests(
    PipelineTesterMixin,
    FirstBlockCacheTesterMixin,
    unittest.TestCase,
):
    pipeline_class = HunyuanImagePipeline
    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt", "negative_prompt"])
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
    test_layerwise_casting = True
    test_group_offloading = True
    test_attention_slicing = False
    supports_dduf = False

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1, guidance_embeds: bool = False):
        torch.manual_seed(0)
        transformer = HunyuanImageTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=4,
            attention_head_dim=8,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=1,
            patch_size=(1, 1),
            guidance_embeds=guidance_embeds,
            text_embed_dim=32,
            text_embed_2_dim=32,
            rope_axes_dim=(4, 4),
        )

        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanImage(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64, 64, 64),
            layers_per_block=1,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            sample_size=128,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        if not guidance_embeds:
            torch.manual_seed(0)
            guider = AdaptiveProjectedMixGuidance(adaptive_projected_guidance_start_step=2)
            ocr_guider = AdaptiveProjectedMixGuidance(adaptive_projected_guidance_start_step=3)
        else:
            guider = None
            ocr_guider = None
        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [2, 2, 4],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
            },
            hidden_size=32,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        torch.manual_seed(0)
        t5_config = T5Config(
            d_model=32,
            d_kv=4,
            d_ff=16,
            num_layers=2,
            num_heads=2,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            vocab_size=256,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            is_encoder_decoder=False,
            use_cache=False,
            tie_word_embeddings=False,
        )
        text_encoder_2 = T5EncoderModel(t5_config)
        tokenizer_2 = ByT5Tokenizer()

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "guider": guider,
            "ocr_guider": ocr_guider,
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
            "num_inference_steps": 5,
            "height": 16,
            "width": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 16, 16))

        expected_slice_np = np.array(
            [0.6252659, 0.51482046, 0.60799813, 0.59267783, 0.488082, 0.5857634, 0.523781, 0.58028054, 0.5674121]
        )
        output_slice = generated_image[0, -3:, -3:].flatten().cpu().numpy()

        self.assertTrue(
            np.abs(output_slice - expected_slice_np).max() < 1e-3,
            f"output_slice: {output_slice}, expected_slice_np: {expected_slice_np}",
        )

    def test_inference_guider(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        pipe.guider = pipe.guider.new(guidance_scale=1000)
        pipe.ocr_guider = pipe.ocr_guider.new(guidance_scale=1000)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 16, 16))

        expected_slice_np = np.array(
            [0.61494756, 0.49616697, 0.60327923, 0.6115793, 0.49047345, 0.56977504, 0.53066164, 0.58880305, 0.5570612]
        )
        output_slice = generated_image[0, -3:, -3:].flatten().cpu().numpy()

        self.assertTrue(
            np.abs(output_slice - expected_slice_np).max() < 1e-3,
            f"output_slice: {output_slice}, expected_slice_np: {expected_slice_np}",
        )

    def test_inference_with_distilled_guidance(self):
        device = "cpu"

        components = self.get_dummy_components(guidance_embeds=True)
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["distilled_guidance_scale"] = 3.5
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 16, 16))

        expected_slice_np = np.array(
            [0.63667065, 0.5187377, 0.66757566, 0.6320319, 0.4913387, 0.54813194, 0.5335031, 0.5736143, 0.5461346]
        )
        output_slice = generated_image[0, -3:, -3:].flatten().cpu().numpy()

        self.assertTrue(
            np.abs(output_slice - expected_slice_np).max() < 1e-3,
            f"output_slice: {output_slice}, expected_slice_np: {expected_slice_np}",
        )

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        generator_device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        # Without tiling
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(tile_sample_min_size=96)
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        self.assertLess(
            (to_np(output_without_tiling) - to_np(output_with_tiling)).max(),
            expected_diff_max,
            "VAE tiling should not affect the inference results",
        )

    @unittest.skip("TODO: Test not supported for now because needs to be adjusted to work with guiders.")
    def test_encode_prompt_works_in_isolation(self):
        pass
