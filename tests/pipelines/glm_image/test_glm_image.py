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
                0.5796329,  0.5005878,  0.45881274, 0.45331675, 0.43688118, 0.4899527, 0.54017603, 0.50983673, 0.3387968,  0.38074082, 0.29942477, 0.33733928, 0.3672544,  0.38462338, 0.40991822, 0.46641728
            ]
        )
        # fmt: on

        self.assertEqual(image.shape, (3, 32, 32))
        self.assertTrue(np.allclose(expected_slice, generated_slice, atol=1e-4, rtol=1e-4))

    @unittest.skip("Not supported.")
    def test_inference_batch_single_identical(self):
        # GLM-Image has batch_size=1 constraint due to AR model
        pass

    @unittest.skip("Not supported.")
    def test_inference_batch_consistent(self):
        # GLM-Image has batch_size=1 constraint due to AR model
        pass

    @unittest.skip("Not supported.")
    def test_num_images_per_prompt(self):
        # GLM-Image has batch_size=1 constraint due to AR model
        pass

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
