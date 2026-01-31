# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import gc
import os
import unittest

import numpy as np
import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImageInpaintPipeline,
    ZImageTransformer2DModel,
)
from diffusers.utils.testing_utils import floats_tensor

from ...testing_utils import torch_device
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin, to_np


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
# Cannot use enable_full_determinism() which sets it to True
# Note: Z-Image does not support FP16 inference due to complex64 RoPE embeddings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False


class ZImageInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ZImageInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset(["image", "mask_image"])
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "strength",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def setUp(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=16,
            dim=32,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            norm_eps=1e-5,
            qk_norm=True,
            cap_feat_dim=16,
            rope_theta=256.0,
            t_scale=1000.0,
            axes_dims=[8, 4, 4],
            axes_lens=[256, 32, 32],
        )
        # `x_pad_token` and `cap_pad_token` are initialized with `torch.empty` which contains
        # uninitialized memory. Set them to known values for deterministic test behavior.
        with torch.no_grad():
            transformer.x_pad_token.copy_(torch.ones_like(transformer.x_pad_token.data))
            transformer.cap_pad_token.copy_(torch.ones_like(transformer.cap_pad_token.data))

        torch.manual_seed(0)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64],
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=32,
            sample_size=32,
            scaling_factor=0.3611,
            shift_factor=0.1159,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

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
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        import random

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        # Create mask: 1 = inpaint region, 0 = preserve region
        mask_image = torch.zeros((1, 1, 32, 32), device=device)
        mask_image[:, :, 8:24, 8:24] = 1.0  # Inpaint center region

        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "image": image,
            "mask_image": mask_image,
            "strength": 1.0,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
            "height": 32,
            "width": 32,
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
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (32, 32, 3))

    def test_inference_batch_single_identical(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-1)

    def test_num_images_per_prompt(self):
        import inspect

        sig = inspect.signature(self.pipeline_class.__call__)

        if "num_images_per_prompt" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        if test_max_difference:
            max_diff1 = np.abs(to_np(output_with_slicing1) - to_np(output_without_slicing)).max()
            max_diff2 = np.abs(to_np(output_with_slicing2) - to_np(output_without_slicing)).max()
            self.assertLess(
                max(max_diff1, max_diff2),
                expected_max_diff,
                "Attention slicing should not affect the inference results",
            )

    def test_vae_tiling(self, expected_diff_max: float = 0.7):
        import random

        generator_device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        # Without tiling
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        # Generate a larger image for the input
        inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(0)).to("cpu")
        # Generate a larger mask for the input
        mask = torch.zeros((1, 1, 128, 128), device="cpu")
        mask[:, :, 32:96, 32:96] = 1.0
        inputs["mask_image"] = mask
        output_without_tiling = pipe(**inputs)[0]

        # With tiling (standard AutoencoderKL doesn't accept parameters)
        pipe.vae.enable_tiling()
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(0)).to("cpu")
        inputs["mask_image"] = mask
        output_with_tiling = pipe(**inputs)[0]

        self.assertLess(
            (to_np(output_without_tiling) - to_np(output_with_tiling)).max(),
            expected_diff_max,
            "VAE tiling should not affect the inference results",
        )

    def test_pipeline_with_accelerator_device_map(self, expected_max_difference=1e-3):
        # Z-Image RoPE embeddings (complex64) have slightly higher numerical tolerance
        # Inpainting mask blending adds additional numerical variance
        super().test_pipeline_with_accelerator_device_map(expected_max_difference=expected_max_difference)

    def test_group_offloading_inference(self):
        # Block-level offloading conflicts with RoPE cache. Pipeline-level offloading (tested separately) works fine.
        self.skipTest("Using test_pipeline_level_group_offloading_inference instead")

    def test_save_load_float16(self, expected_max_diff=1e-2):
        # Z-Image does not support FP16 due to complex64 RoPE embeddings
        self.skipTest("Z-Image does not support FP16 inference")

    def test_float16_inference(self, expected_max_diff=5e-2):
        # Z-Image does not support FP16 due to complex64 RoPE embeddings
        self.skipTest("Z-Image does not support FP16 inference")

    def test_strength_parameter(self):
        """Test that strength parameter affects the output correctly."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Test with different strength values
        inputs_low_strength = self.get_dummy_inputs(device)
        inputs_low_strength["strength"] = 0.2

        inputs_high_strength = self.get_dummy_inputs(device)
        inputs_high_strength["strength"] = 0.8

        # Both should complete without errors
        output_low = pipe(**inputs_low_strength).images[0]
        output_high = pipe(**inputs_high_strength).images[0]

        # Outputs should be different (different amount of transformation)
        self.assertFalse(np.allclose(output_low, output_high, atol=1e-3))

    def test_invalid_strength(self):
        """Test that invalid strength values raise appropriate errors."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)

        inputs = self.get_dummy_inputs(device)

        # Test strength < 0
        inputs["strength"] = -0.1
        with self.assertRaises(ValueError):
            pipe(**inputs)

        # Test strength > 1
        inputs["strength"] = 1.5
        with self.assertRaises(ValueError):
            pipe(**inputs)

    def test_mask_inpainting(self):
        """Test that the mask properly controls which regions are inpainted."""
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Generate with full mask (inpaint everything)
        inputs_full = self.get_dummy_inputs(device)
        inputs_full["mask_image"] = torch.ones((1, 1, 32, 32), device=device)

        # Generate with no mask (preserve everything)
        inputs_none = self.get_dummy_inputs(device)
        inputs_none["mask_image"] = torch.zeros((1, 1, 32, 32), device=device)

        # Both should complete without errors
        output_full = pipe(**inputs_full).images[0]
        output_none = pipe(**inputs_none).images[0]

        # Outputs should be different (full inpaint vs preserve)
        self.assertFalse(np.allclose(output_full, output_none, atol=1e-3))
