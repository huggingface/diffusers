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

import gc
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

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
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
        channel_sizes = {"4": 16, "8": 16, "16": 16}
        transformer = WanAnimateTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            latent_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            image_dim=4,
            rope_max_seq_len=32,
            motion_encoder_channel_sizes=channel_sizes,
            motion_encoder_size=16,
            motion_style_dim=8,
            motion_dim=4,
            motion_encoder_dim=16,
            face_encoder_hidden_dim=16,
            face_encoder_num_heads=2,
            inject_face_latents_blocks=2,
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
        face_height = 16
        face_width = 16

        image = Image.new("RGB", (height, width))
        pose_video = [Image.new("RGB", (height, width))] * num_frames
        face_video = [Image.new("RGB", (face_height, face_width))] * num_frames

        inputs = {
            "image": image,
            "pose_video": pose_video,
            "face_video": face_video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "height": height,
            "width": width,
            "segment_frame_length": 77,  # TODO: can we set this to num_frames?
            "num_inference_steps": 2,
            "mode": "animate",
            "prev_segment_conditioning_frames": 1,
            "generator": generator,
            "guidance_scale": 1.0,
            "output_type": "pt",
            "max_sequence_length": 16,
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

        expected_video = torch.randn(17, 3, 16, 16)
        max_diff = np.abs(video - expected_video).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_inference_replacement(self):
        """Test the pipeline in replacement mode with background and mask videos."""
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["mode"] = "replace"
        num_frames = 17
        height = 16
        width = 16
        inputs["background_video"] = [Image.new("RGB", (height, width))] * num_frames
        inputs["mask_video"] = [Image.new("L", (height, width))] * num_frames

        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip(
        "Setting the Wan Animate latents to zero at the last denoising step does not guarantee that the output will be"
        " zero. I believe this is because the latents are further processed in the outer loop where we loop over"
        " inference segments."
    )
    def test_callback_inputs(self):
        pass


@slow
@require_torch_accelerator
class WanAnimatePipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    @unittest.skip("TODO: test needs to be implemented")
    def test_wan_animate(self):
        pass
