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

from diffusers.modular_pipelines import (
    HeliosAutoBlocks,
    HeliosModularPipeline,
    HeliosPyramidAutoBlocks,
    HeliosPyramidModularPipeline,
)

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


HELIOS_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.set_timesteps", "HeliosSetTimestepsStep"),
        ("denoise.chunk_denoise", "HeliosChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
    "image2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("vae_encoder", "HeliosImageVaeEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.additional_inputs", "HeliosAdditionalInputsStep"),
        ("denoise.add_noise_image", "HeliosAddNoiseToImageLatentsStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.seed_history", "HeliosI2VSeedHistoryStep"),
        ("denoise.set_timesteps", "HeliosSetTimestepsStep"),
        ("denoise.chunk_denoise", "HeliosI2VChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
    "video2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("vae_encoder", "HeliosVideoVaeEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.additional_inputs", "HeliosAdditionalInputsStep"),
        ("denoise.add_noise_video", "HeliosAddNoiseToVideoLatentsStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.seed_history", "HeliosV2VSeedHistoryStep"),
        ("denoise.set_timesteps", "HeliosSetTimestepsStep"),
        ("denoise.chunk_denoise", "HeliosI2VChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
}


class TestHeliosModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = HeliosModularPipeline
    pipeline_blocks_class = HeliosAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-helios-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    output_name = "videos"
    expected_workflow_blocks = HELIOS_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


HELIOS_PYRAMID_WORKFLOWS = {
    "text2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "HeliosPyramidChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
    "image2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("vae_encoder", "HeliosImageVaeEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.additional_inputs", "HeliosAdditionalInputsStep"),
        ("denoise.add_noise_image", "HeliosAddNoiseToImageLatentsStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.seed_history", "HeliosI2VSeedHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "HeliosPyramidI2VChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
    "video2video": [
        ("text_encoder", "HeliosTextEncoderStep"),
        ("vae_encoder", "HeliosVideoVaeEncoderStep"),
        ("denoise.input", "HeliosTextInputStep"),
        ("denoise.additional_inputs", "HeliosAdditionalInputsStep"),
        ("denoise.add_noise_video", "HeliosAddNoiseToVideoLatentsStep"),
        ("denoise.prepare_history", "HeliosPrepareHistoryStep"),
        ("denoise.seed_history", "HeliosV2VSeedHistoryStep"),
        ("denoise.pyramid_chunk_denoise", "HeliosPyramidI2VChunkDenoiseStep"),
        ("decode", "HeliosDecodeStep"),
    ],
}


class TestHeliosPyramidModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = HeliosPyramidModularPipeline
    pipeline_blocks_class = HeliosPyramidAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-helios-pyramid-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["pyramid_num_inference_steps_list", "num_videos_per_prompt", "latents"])
    output_name = "videos"
    expected_workflow_blocks = HELIOS_PYRAMID_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "pyramid_num_inference_steps_list": [2, 2],
            "height": 64,
            "width": 64,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        # Pyramid pipeline injects noise at each stage, so batch vs single can differ more
        super().test_inference_batch_single_identical(expected_max_diff=5e-1)

    @pytest.mark.skip(reason="Pyramid multi-stage noise makes offload comparison unreliable with tiny models")
    def test_components_auto_cpu_offload_inference_consistent(self):
        pass

    @pytest.mark.skip(reason="Pyramid multi-stage noise makes save/load comparison unreliable with tiny models")
    def test_save_from_pretrained(self):
        pass

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass
