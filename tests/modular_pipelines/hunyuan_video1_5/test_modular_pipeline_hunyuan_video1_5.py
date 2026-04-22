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

from diffusers.modular_pipelines import HunyuanVideo15AutoBlocks, HunyuanVideo15ModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


HUNYUANVIDEO15_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "HunyuanVideo15TextEncoderStep"),
        ("denoise.input", "HunyuanVideo15TextInputStep"),
        ("denoise.set_timesteps", "HunyuanVideo15SetTimestepsStep"),
        ("denoise.prepare_latents", "HunyuanVideo15PrepareLatentsStep"),
        ("denoise.denoise", "HunyuanVideo15DenoiseStep"),
        ("decode", "HunyuanVideo15VaeDecoderStep"),
    ],
    "image2video": [
        ("text_encoder", "HunyuanVideo15TextEncoderStep"),
        ("vae_encoder", "HunyuanVideo15VaeEncoderStep"),
        ("image_encoder", "HunyuanVideo15ImageEncoderStep"),
        ("denoise.input", "HunyuanVideo15TextInputStep"),
        ("denoise.set_timesteps", "HunyuanVideo15SetTimestepsStep"),
        ("denoise.prepare_latents", "HunyuanVideo15PrepareLatentsStep"),
        ("denoise.prepare_i2v_latents", "HunyuanVideo15Image2VideoPrepareLatentsStep"),
        ("denoise.denoise", "HunyuanVideo15Image2VideoDenoiseStep"),
        ("decode", "HunyuanVideo15VaeDecoderStep"),
    ],
}


class TestHunyuanVideo15ModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = HunyuanVideo15ModularPipeline
    pipeline_blocks_class = HunyuanVideo15AutoBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-hunyuanvideo1_5-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    expected_workflow_blocks = HUNYUANVIDEO15_WORKFLOWS
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
            "output_type": "pt",
        }
        return inputs

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(reason="VAE causal attention mask does not support batch>1 decode")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="VAE causal attention mask does not support batch>1 decode")
    def test_inference_batch_single_identical(self):
        pass

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=0.1)
