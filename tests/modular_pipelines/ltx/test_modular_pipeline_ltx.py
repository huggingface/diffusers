# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers.modular_pipelines import LTXAutoBlocks, LTXModularPipeline

from ...testing_utils import torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


LTX_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "LTXTextEncoderStep"),
        ("denoise.input", "LTXTextInputStep"),
        ("denoise.set_timesteps", "LTXSetTimestepsStep"),
        ("denoise.prepare_latents", "LTXPrepareLatentsStep"),
        ("denoise.denoise", "LTXDenoiseStep"),
        ("denoise.unpack_latents", "LTXUnpackLatentsStep"),
        ("decode", "LTXVaeDecoderStep"),
    ],
    "image2video": [
        ("text_encoder", "LTXTextEncoderStep"),
        ("vae_encoder", "LTXVaeEncoderStep"),
        ("denoise.input", "LTXTextInputStep"),
        ("denoise.set_timesteps", "LTXSetTimestepsStep"),
        ("denoise.prepare_latents", "LTXPrepareLatentsStep"),
        ("denoise.prepare_i2v_latents", "LTXImage2VideoPrepareLatentsStep"),
        ("denoise.denoise", "LTXImage2VideoDenoiseStep"),
        ("denoise.unpack_latents", "LTXUnpackLatentsStep"),
        ("decode", "LTXVaeDecoderStep"),
    ],
}


class LTXModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = LTXModularPipeline
    pipeline_blocks_class = LTXAutoBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-ltx-modular-pipe"
    expected_latents_shape = (1, 8, 5, 16, 16)
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    expected_workflow_blocks = LTX_WORKFLOWS
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs


class TestLTXModularPipelineFast(LTXModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_decode_accepts_packed_latents_with_deprecation(self):
        # The old packed `[B, S, C]` form is still accepted by the decode step for a deprecation window.
        decoder = self.pipeline_blocks_class().sub_blocks["decode"].init_pipeline(self.pretrained_model_name_or_path)
        decoder.load_components(dtype=torch.float32)
        decoder.to(torch_device)

        # The 9-frame 32x32 dummy video is a `[1, 8, 5, 16, 16]` latent, packed to 1280 tokens of 8 channels.
        packed_latents = torch.randn(1, 1280, 8, generator=self.get_generator()).to(torch_device)
        with pytest.warns(FutureWarning, match="packed latents"):
            videos = decoder(
                latents=packed_latents,
                height=32,
                width=32,
                num_frames=9,
                batch_size=1,
                dtype=torch.float32,
                output_type="pt",
                output="videos",
            )

        assert tuple(videos.shape[-2:]) == (32, 32)

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


class TestLTXModularPipelineLoading(LTXModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestLTXModularPipelineWorkflow(LTXModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestLTXModularPipelineMemory(LTXModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
