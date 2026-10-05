# Copyright 2025 The Kandinsky Team and The HuggingFace Team.
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

import torch

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    Kandinsky6SRLatentUpscalerBank,
    Kandinsky6SRPipeline,
    Kandinsky6SRTransformer3DModel,
    Kandinsky6SRVAE,
)

from ...testing_utils import torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class Kandinsky6SRPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Kandinsky6SRPipeline
    required_input_params_in_call_signature = frozenset(["video", "resolution_scale", "num_inference_steps"])
    batch_input_params = frozenset(["video"])
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])
    # (num_frames, channels, height, width) for output_type="pt": the 5x16x32 input video upscaled 2x.
    output_shape = (5, 3, 32, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Kandinsky6SRTransformer3DModel(
            in_visual_dim=4,
            out_visual_dim=4,
            time_dim=16,
            patch_size=(1, 1, 1),
            model_dim=24,
            ff_dim=32,
            num_visual_blocks=2,
            axes_dims=(4, 4, 4),
            # Trained tile resolutions. NABLA needs each tile's latent token grid divisible by 8, so every size is a
            # multiple of 32 (VAE spatial factor 4 times 8). The 2x route tiles the 16x32 test video at (16, 32).
            tile_sizes=((32, 32), (32, 64), (64, 32)),
        )

        torch.manual_seed(0)
        # Three levels give a 4x spatial factor; both non-final levels compress time for a 4x temporal factor.
        vae = Kandinsky6SRVAE(
            latent_channels=4,
            encoder_block_out_channels=(4, 8, 8),
            decoder_block_out_channels=(4, 8, 8),
            layers_per_block=1,
            temporal_compression_ratio=4,
            temporal_compression_start_level=0,
        )

        torch.manual_seed(0)
        latent_upscaler = Kandinsky6SRLatentUpscalerBank(
            in_channels=4,
            stage_channels=(8, 8, 4),
            num_pre_blocks=1,
            num_mid_blocks=1,
            num_post_blocks=1,
            num_x2_adapter_blocks=1,
            scales=(2, 4),
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.5)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "latent_upscaler": latent_upscaler,
        }

    def get_dummy_inputs(self):
        # A `(num_frames, channels, height, width)` video in [0, 1] with 1 + 4k frames.
        video = torch.rand((5, 3, 16, 32), generator=self.get_generator(0))
        return {
            "video": video,
            "resolution_scale": 2,
            "num_inference_steps": 2,
            "generator": self.get_generator(0),
            "min_overlap": 0.2,
            "tiles_batch_size": 4,
            "output_type": "pt",
        }


class TestKandinsky6SRPipeline(Kandinsky6SRPipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky6_sr_scales(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        # The 2.25x route rounds the 1.125x pre-upscaled size to the VAE spatial factor (4 here): 16 -> 16, 32 -> 36.
        for resolution_scale, expected in ((2, (32, 64)), (2.25, (32, 72)), (4, (64, 128))):
            inputs = self.get_dummy_inputs()
            inputs["resolution_scale"] = resolution_scale
            output = pipe(**inputs)
            assert output.frames.shape == (1, 5, 3, *expected), f"unexpected shape for scale {resolution_scale}"
            assert not torch.isnan(output.frames).any()

    def test_kandinsky6_sr_without_latent_upscaler(self):
        # Without the latent upscaler the pixel tiles are bilinearly upscaled and encoded instead.
        components = self.get_dummy_components()
        components["latent_upscaler"] = None
        pipe = self.pipeline_class(**components).to(torch_device)
        output = pipe(**self.get_dummy_inputs())
        assert output.frames.shape == (1, *self.output_shape)
        assert not torch.isnan(output.frames).any()

    def test_kandinsky6_sr_rejects_wrong_frame_count(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["video"] = inputs["video"][:4]
        try:
            pipe(**inputs)
        except ValueError as error:
            assert "frames" in str(error)
        else:
            raise AssertionError("expected a ValueError for a video without 1 + 4k frames")


class TestKandinsky6SRPipelineMemory(Kandinsky6SRPipelineTesterConfig, MemoryTesterMixin):
    pass
