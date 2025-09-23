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
import pytest
import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class QwenImageEditPlusPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = QwenImageEditPlusPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["prompt", "image"])
    image_params = frozenset(["image"])
    image_latents_params = frozenset(["latents"])
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
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        tiny_ckpt_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=3,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            hidden_size=16,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained(tiny_ckpt_id)

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": Qwen2VLProcessor.from_pretrained(tiny_ckpt_id),
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = Image.new("RGB", (32, 32))
        inputs = {
            "prompt": "dance monkey",
            "image": [image, image],
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "true_cfg_scale": 1.0,
            "height": 32,
            "width": 32,
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
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 32, 32))

        # fmt: off
        expected_slice = torch.tensor([[0.5637, 0.6341, 0.6001, 0.5620, 0.5794, 0.5498, 0.5757, 0.6389, 0.4174, 0.3597, 0.5649, 0.4894, 0.4969, 0.5255, 0.4083, 0.4986]])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

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
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        self.assertLess(
            (to_np(output_without_tiling) - to_np(output_with_tiling)).max(),
            expected_diff_max,
            "VAE tiling should not affect the inference results",
        )

    @pytest.mark.xfail(condition=True, reason="Preconfigured embeddings need to be revisited.", strict=True)
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol, rtol)

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_num_images_per_prompt():
        super().test_num_images_per_prompt()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_consistent():
        super().test_inference_batch_consistent()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_single_identical():
        super().test_inference_batch_single_identical()
