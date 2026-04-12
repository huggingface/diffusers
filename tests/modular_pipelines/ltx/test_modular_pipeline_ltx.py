# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers.modular_pipelines import LTXAutoBlocks, LTXModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


LTX_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "LTXTextEncoderStep"),
        ("denoise.input", "LTXTextInputStep"),
        ("denoise.set_timesteps", "LTXSetTimestepsStep"),
        ("denoise.prepare_latents", "LTXPrepareLatentsStep"),
        ("denoise.denoise", "LTXDenoiseStep"),
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
        ("decode", "LTXVaeDecoderStep"),
    ],
}


class TestLTXModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = LTXModularPipeline
    pipeline_blocks_class = LTXAutoBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-ltx-modular-pipe"

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

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass
