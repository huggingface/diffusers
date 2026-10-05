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

from diffusers.modular_pipelines import (
    HeliosAutoBlocks,
    HeliosModularPipeline,
    HeliosPyramidAutoBlocks,
    HeliosPyramidModularPipeline,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


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


class HeliosModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
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


class HeliosChunkCallbackTesterMixin:
    def test_step_callback_chunks_and_stages(self):
        pipe = self.get_pipeline().to("cpu")
        inputs = self.get_dummy_inputs()
        inputs.update(num_frames=18, num_latent_frames_per_chunk=3)
        steps_per_chunk = sum(inputs.get("pyramid_num_inference_steps_list", [inputs.get("num_inference_steps")]))
        steps = []

        def record(pipeline, step, timestep, tensors):
            steps.append(step)
            return tensors

        full = pipe(**inputs, callback_on_step_end=record, output=["videos", "latent_chunks"])
        assert steps == list(range(2 * steps_per_chunk))
        assert len(full["latent_chunks"]) == 2
        stop_steps = range(0, 2 * steps_per_chunk, 2)
        for stop_step in stop_steps:
            steps.clear()
            inputs["generator"] = self.get_generator()

            def stop(pipeline, step, timestep, tensors):
                steps.append(step)
                if step == stop_step:
                    pipeline.interrupt = True
                return tensors

            partial = pipe(**inputs, callback_on_step_end=stop, output=["videos", "latent_chunks"])
            assert steps == list(range(stop_step + 1))
            chunks = stop_step // steps_per_chunk + 1
            assert len(partial["latent_chunks"]) == chunks
            assert partial["videos"].shape == (1, chunks * 8 + 1, 3, inputs["height"], inputs["width"])
            assert torch.isfinite(partial["videos"]).all()


class TestHeliosModularPipelineFast(
    HeliosModularPipelineTesterConfig, HeliosChunkCallbackTesterMixin, ModularPipelineTesterMixin
):
    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


class TestHeliosModularPipelineLoading(HeliosModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestHeliosModularPipelineWorkflow(HeliosModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestHeliosModularPipelineMemory(HeliosModularPipelineTesterConfig, ModularMemoryTesterMixin):
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


class HeliosPyramidModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
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


class TestHeliosPyramidModularPipelineFast(
    HeliosPyramidModularPipelineTesterConfig, HeliosChunkCallbackTesterMixin, ModularPipelineTesterMixin
):
    def test_inference_batch_single_identical(self):
        # Pyramid pipeline injects noise at each stage, so batch vs single can differ more
        super().test_inference_batch_single_identical(expected_max_diff=5e-1)

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


class TestHeliosPyramidModularPipelineLoading(HeliosPyramidModularPipelineTesterConfig, ModularLoadingTesterMixin):
    @pytest.mark.skip(reason="Pyramid multi-stage noise makes save/load comparison unreliable with tiny models")
    def test_save_from_pretrained(self):
        pass


class TestHeliosPyramidModularPipelineWorkflow(HeliosPyramidModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestHeliosPyramidModularPipelineMemory(HeliosPyramidModularPipelineTesterConfig, ModularMemoryTesterMixin):
    @pytest.mark.skip(reason="Pyramid multi-stage noise makes offload comparison unreliable with tiny models")
    def test_components_auto_cpu_offload_inference_consistent(self):
        pass
