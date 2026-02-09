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
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, GlmImagePipeline, GlmImageTransformer2DModel
from diffusers.utils import is_transformers_version

from ...testing_utils import enable_full_determinism, require_torch_accelerator, require_transformers_version_greater
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


if is_transformers_version(">=", "5.0.0.dev0"):
    from transformers import GlmImageConfig, GlmImageForConditionalGeneration, GlmImageProcessor


enable_full_determinism()


@require_transformers_version_greater("4.57.4")
@require_torch_accelerator
class GlmImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = GlmImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "negative_prompt"}
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
    test_attention_slicing = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        glm_config = GlmImageConfig(
            text_config={
                "vocab_size": 168064,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "vision_vocab_size": 128,
                "rope_parameters": {"mrope_section": (4, 2, 2)},
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "num_heads": 2,
                "image_size": 32,
                "patch_size": 8,
                "intermediate_size": 32,
            },
            vq_config={"embed_dim": 32, "num_embeddings": 128, "latent_channels": 32},
        )

        torch.manual_seed(0)
        vision_language_encoder = GlmImageForConditionalGeneration(glm_config)

        processor = GlmImageProcessor.from_pretrained("zai-org/GLM-Image", subfolder="processor")

        torch.manual_seed(0)
        # For GLM-Image, the relationship between components must satisfy:
        # patch_size × vae_scale_factor = 16 (since AR tokens are upsampled 2× from d32)
        transformer = GlmImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            text_embed_dim=text_encoder.config.hidden_size,
            time_embed_dim=16,
            condition_dim=8,
            prior_vq_quantizer_codebook_size=128,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=(4, 8, 16, 16),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=4,
            sample_size=128,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        components = {
            "tokenizer": tokenizer,
            "processor": processor,
            "text_encoder": text_encoder,
            "vision_language_encoder": vision_language_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": scheduler,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        height, width = 32, 32

        inputs = {
            "prompt": "A photo of a cat",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": height,
            "width": width,
            "max_sequence_length": 16,
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
        image = pipe(**inputs).images[0]
        generated_slice = image.flatten()
        generated_slice = np.concatenate([generated_slice[:8], generated_slice[-8:]])

        # fmt: off
        expected_slice = np.array(
            [
                0.5849247, 0.50278825, 0.45747858, 0.45895284, 0.43804976, 0.47044256, 0.5239665, 0.47904694, 0.3323419, 0.38725388, 0.28505728, 0.3161863, 0.35026982, 0.37546024, 0.4090118, 0.46629113
            ]
        )
        # fmt: on

        self.assertEqual(image.shape, (3, 32, 32))
        self.assertTrue(np.allclose(expected_slice, generated_slice, atol=1e-4, rtol=1e-4))

    def test_inference_batch_single_identical(self):
        """Test that batch=1 produces consistent results with the same seed."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Run twice with same seed
        inputs1 = self.get_dummy_inputs(device, seed=42)
        inputs2 = self.get_dummy_inputs(device, seed=42)

        image1 = pipe(**inputs1).images[0]
        image2 = pipe(**inputs2).images[0]

        self.assertTrue(torch.allclose(image1, image2, atol=1e-4))

    def test_inference_batch_multiple_prompts(self):
        """Test batch processing with multiple prompts."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        height, width = 32, 32

        inputs = {
            "prompt": ["A photo of a cat", "A photo of a dog"],
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": height,
            "width": width,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

        images = pipe(**inputs).images

        # Should return 2 images
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].shape, (3, 32, 32))
        self.assertEqual(images[1].shape, (3, 32, 32))

    def test_num_images_per_prompt(self):
        """Test generating multiple images per prompt."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        height, width = 32, 32

        inputs = {
            "prompt": "A photo of a cat",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": height,
            "width": width,
            "max_sequence_length": 16,
            "output_type": "pt",
            "num_images_per_prompt": 2,
        }

        images = pipe(**inputs).images

        # Should return 2 images for single prompt
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].shape, (3, 32, 32))
        self.assertEqual(images[1].shape, (3, 32, 32))

    def test_batch_with_num_images_per_prompt(self):
        """Test batch prompts with num_images_per_prompt > 1."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        height, width = 32, 32

        inputs = {
            "prompt": ["A photo of a cat", "A photo of a dog"],
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": height,
            "width": width,
            "max_sequence_length": 16,
            "output_type": "pt",
            "num_images_per_prompt": 2,
        }

        images = pipe(**inputs).images

        # Should return 4 images (2 prompts × 2 images per prompt)
        self.assertEqual(len(images), 4)

    def test_prompt_with_prior_token_ids(self):
        """Test that prompt and prior_token_ids can be provided together.

        When both are given, the AR generation step is skipped (prior_token_ids is used
        directly) and prompt is used to generate prompt_embeds via the glyph encoder.
        """
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        height, width = 32, 32

        # Step 1: Run with prompt only to get prior_token_ids from AR model
        generator = torch.Generator(device=device).manual_seed(0)
        prior_token_ids, _, _ = pipe.generate_prior_tokens(
            prompt="A photo of a cat",
            height=height,
            width=width,
            device=torch.device(device),
            generator=torch.Generator(device=device).manual_seed(0),
        )

        # Step 2: Run with both prompt and prior_token_ids — should not raise
        generator = torch.Generator(device=device).manual_seed(0)
        inputs_both = {
            "prompt": "A photo of a cat",
            "prior_token_ids": prior_token_ids,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": height,
            "width": width,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        images = pipe(**inputs_both).images
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (3, 32, 32))

    def test_check_inputs_rejects_invalid_combinations(self):
        """Test that check_inputs correctly rejects invalid input combinations."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)

        height, width = 32, 32

        # Neither prompt nor prior_token_ids → error
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prompt_embeds=torch.randn(1, 16, 32),
            )

        # prior_token_ids alone without prompt or prompt_embeds → error
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt=None,
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prior_token_ids=torch.randint(0, 100, (1, 64)),
            )

        # prompt + prompt_embeds together → error
        with self.assertRaises(ValueError):
            pipe.check_inputs(
                prompt="A cat",
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prompt_embeds=torch.randn(1, 16, 32),
            )

    @unittest.skip("Needs to be revisited.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Needs to be revisited.")
    def test_pipeline_level_group_offloading_inference(self):
        pass

    @unittest.skip(
        "Follow set of tests are relaxed because this pipeline doesn't guarantee same outputs for the same inputs in consecutive runs."
    )
    def test_dict_tuple_outputs_equivalent(self):
        pass

    @unittest.skip("Skipped")
    def test_cpu_offload_forward_pass_twice(self):
        pass

    @unittest.skip("Skipped")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    @unittest.skip("Skipped")
    def test_float16_inference(self):
        pass

    @unittest.skip("Skipped")
    def test_save_load_float16(self):
        pass

    @unittest.skip("Skipped")
    def test_save_load_local(self):
        pass
