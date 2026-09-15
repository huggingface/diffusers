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

import pytest
import torch

from diffusers import AutoencoderKLLTXVideo, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LTXLatentUpsamplePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LTXLatentUpsamplePipeline
    required_input_params_in_call_signature = frozenset(["video", "height", "width", "latents"])
    batch_input_params = frozenset(["video"])
    output_shape = (5, 3, 32, 32)
    # This pipeline takes a video rather than a prompt, so it has neither `num_images_per_prompt` nor
    # `num_inference_steps` — upsampling is a single forward pass through the latent upsampler.
    optional_input_params = frozenset(["generator", "latents", "output_type", "return_dict"])

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

        return {
            "vae": vae,
            "latent_upsampler": latent_upsampler,
        }

    def get_dummy_inputs(self):
        generator = self.get_generator(0)
        video = torch.randn((5, 3, 32, 32), generator=generator)

        return {
            "video": video,
            "generator": generator,
            "height": 16,
            "width": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTXLatentUpsamplePipeline(LTXLatentUpsamplePipelineTesterConfig, PipelineTesterMixin):
    def test_vae_tiling(self, expected_diff_max: float = 0.25):
        pipe = self.get_pipeline().to(torch_device)

        # Without tiling
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )

    # `__call__` documents batched video input as unsupported (`batch_size` is pinned to 1), so the batching
    # tests below have nothing to assert against.
    @pytest.mark.skip("Batched video input is not supported by this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("Batched video input is not supported by this pipeline.")
    def test_inference_batch_single_identical(self):
        pass


class TestLTXLatentUpsamplePipelineMemory(LTXLatentUpsamplePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX upsampler pipeline."""
