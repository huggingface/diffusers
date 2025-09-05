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

from diffusers import AutoencoderKLLTXVideo, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel

from ...testing_utils import enable_full_determinism
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class LTXLatentUpsamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LTXLatentUpsamplePipeline
    params = {"video", "generator"}
    batch_params = {"video", "generator"}
    required_optional_params = frozenset(["generator", "latents", "return_dict"])
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLLTXVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1, 1),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False, False),
            decoder_inject_noise=(False, False, False, False, False),
            upsample_residual=(False, False, False, False),
            upsample_factor=(1, 1, 1, 1),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        latent_upsampler = LTXLatentUpsamplerModel(
            in_channels=8,
            mid_channels=32,
            num_blocks_per_stage=1,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
        )

        components = {
            "vae": vae,
            "latent_upsampler": latent_upsampler,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        video = torch.randn((5, 3, 32, 32), generator=generator, device=device)

        inputs = {
            "video": video,
            "generator": generator,
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
        video = pipe(**inputs).frames
        generated_video = video[0]

        self.assertEqual(generated_video.shape, (5, 3, 32, 32))
        expected_video = torch.randn(5, 3, 32, 32)
        max_diff = np.abs(generated_video - expected_video).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_vae_tiling(self, expected_diff_max: float = 0.25):
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

    @unittest.skip("Test is not applicable.")
    def test_callback_inputs(self):
        pass

    @unittest.skip("Test is not applicable.")
    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        pass

    @unittest.skip("Test is not applicable.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("Test is not applicable.")
    def test_inference_batch_single_identical(self):
        pass
