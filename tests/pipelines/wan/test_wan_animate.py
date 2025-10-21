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

"""
Tests for WanAnimatePipeline.

This test suite covers:
- Basic inference in animation mode
- Inference with reference images (single and multiple)
- Replacement mode with background and mask videos
- Temporal guidance with different frame counts
- Callback functionality
- Pre-generated embeddings (prompt, negative_prompt, image)
- Pre-generated latents
- Various edge cases and parameter combinations
"""

import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanAnimatePipeline,
    WanAnimateTransformer3DModel,
)

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class WanAnimatePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanAnimatePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
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
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanAnimateTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            image_dim=4,
            pos_embed_seq_len=2 * (4 * 4 + 1),
        )

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=4,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        torch.manual_seed(0)
        image_processor = CLIPImageProcessor(crop_size=4, size=4)

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        num_frames = 17
        height = 16
        width = 16

        pose_video = [Image.new("RGB", (height, width))] * num_frames
        face_video = [Image.new("RGB", (height, width))] * num_frames
        image = Image.new("RGB", (height, width))

        inputs = {
            "image": image,
            "pose_video": pose_video,
            "face_video": face_video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "mode": "animation",
            "num_frames_for_temporal_guidance": 1,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        """Test basic inference in animation mode."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.4523, 0.45198, 0.44872, 0.45326, 0.45211, 0.45258, 0.45344, 0.453, 0.52431, 0.52572, 0.50701, 0.5118, 0.53717, 0.53093, 0.50557, 0.51402]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    def test_inference_with_single_reference_image(self):
        """Test inference with a single reference image for additional context."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["reference_images"] = Image.new("RGB", (16, 16))
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.45247, 0.45214, 0.44874, 0.45314, 0.45171, 0.45299, 0.45428, 0.45317, 0.51378, 0.52658, 0.53361, 0.52303, 0.46204, 0.50435, 0.52555, 0.51342]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    def test_inference_with_multiple_reference_image(self):
        """Test inference with multiple reference images for richer context."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["reference_images"] = [[Image.new("RGB", (16, 16))] * 2]
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.45321, 0.45221, 0.44818, 0.45375, 0.45268, 0.4519, 0.45271, 0.45253, 0.51244, 0.52223, 0.51253, 0.51321, 0.50743, 0.51177, 0.51626, 0.50983]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Errors out because passing multiple prompts at once is not yet supported by this pipeline.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_float16_inference(self):
        pass

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_save_load_float16(self):
        pass

    def test_inference_replacement_mode(self):
        """Test the pipeline in replacement mode with background and mask videos."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["mode"] = "replacement"
        # Create background and mask videos for replacement mode
        num_frames = 17
        height = 16
        width = 16
        inputs["background_video"] = [Image.new("RGB", (height, width))] * num_frames
        inputs["mask_video"] = [Image.new("RGB", (height, width))] * num_frames

        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    def test_inference_with_temporal_guidance_5_frames(self):
        """Test inference with 5 frames for temporal guidance instead of default 1."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_frames_for_temporal_guidance"] = 5
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    def test_inference_with_callback_on_step_end(self):
        """Test that callback functions are called during inference."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        callback_fn_output = {"latents": []}

        def callback_fn(pipe, i, t, callback_kwargs):
            callback_fn_output["latents"].append(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_fn
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]

        output = pipe(**inputs)
        self.assertTrue(len(callback_fn_output["latents"]) > 0)

    def test_inference_with_provided_embeddings(self):
        """Test inference with pre-generated text and image embeddings."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        # Generate embeddings beforehand
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=inputs["prompt"],
            negative_prompt=inputs["negative_prompt"],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=device,
        )

        image_embeds = pipe.encode_image(inputs["image"], device)

        # Remove text prompts and provide embeddings instead
        inputs.pop("prompt")
        inputs.pop("negative_prompt")
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["image_embeds"] = image_embeds

        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    def test_inference_with_provided_latents(self):
        """Test inference with pre-generated latents for reproducibility."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        # Generate random latents
        num_frames = inputs["num_frames"]
        height = inputs["height"]
        width = inputs["width"]
        latent_height = height // pipe.vae_scale_factor_spatial
        latent_width = width // pipe.vae_scale_factor_spatial
        num_latent_frames = num_frames // pipe.vae_scale_factor_temporal + 1

        latents = torch.randn(
            1, 16, num_latent_frames + 1, latent_height, latent_width
        )

        inputs["latents"] = latents
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))
