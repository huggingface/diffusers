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
from transformers import Qwen2Tokenizer, Qwen2VLForConditionalGeneration

from diffusers import (
    AutoencoderKLMagvit,
    EasyAnimateInpaintPipeline,
    EasyAnimateTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class EasyAnimateInpaintPipelineFastTests(unittest.TestCase):
    pipeline_class = EasyAnimateInpaintPipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = EasyAnimateTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            time_embed_dim=2,
            text_embed_dim=16,  # Must match with tiny-random-t5
            num_layers=1,
            sample_width=16,  # latent width: 2 -> final width: 16
            sample_height=16,  # latent height: 2 -> final height: 16
            patch_size=2,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLMagvit(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "SpatialDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
            ),
            up_block_types=(
                "SpatialUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            spatial_group_norm=False,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
        )
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
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        video = torch.rand(1, 3, 5, 16, 16, generator=torch.Generator().manual_seed(seed))
        # Partial mask so the inpainting preservation branch is exercised.
        mask_video = torch.zeros(1, 1, 5, 16, 16)
        mask_video[:, :, :, 8:, :] = 255.0

        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "video": video,
            "mask_video": mask_video,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": 5,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        # When `num_channels_transformer == num_channels_latents`, every denoising step blends
        # `image_latents` back into `latents` through `scheduler.scale_noise`.
        # Regression test for https://github.com/huggingface/diffusers/issues/12646.
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]

        self.assertEqual(generated_video.shape, (5, 3, 16, 16))
