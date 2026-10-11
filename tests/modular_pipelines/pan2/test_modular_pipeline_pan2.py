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

import PIL.Image
import pytest
import torch

from diffusers.modular_pipelines import ModularPipeline, PAN2Blocks, PAN2ModularPipeline

from ...testing_utils import torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


PAN2_WORKFLOWS = {
    "t2v": [
        ("text_encoder", "PAN2TextEncoderStep"),
        ("denoise.input", "PAN2TextInputStep"),
        ("denoise.set_timesteps", "PAN2SetTimestepsStep"),
        ("denoise.prepare_latents", "PAN2PrepareLatentsStep"),
        ("denoise.prepare_condition_latents", "PAN2PrepareConditionLatentsStep"),
        ("denoise.denoise", "PAN2DenoiseStep"),
        ("decode", "PAN2VaeDecoderStep"),
    ],
    "i2v": [
        ("text_encoder", "PAN2TextEncoderStep"),
        ("vae_encoder", "PAN2VaeEncoderStep"),
        ("denoise.input", "PAN2TextInputStep"),
        ("denoise.set_timesteps", "PAN2SetTimestepsStep"),
        ("denoise.prepare_latents", "PAN2PrepareLatentsStep"),
        ("denoise.prepare_condition_latents", "PAN2Image2VideoPrepareConditionLatentsStep"),
        ("denoise.denoise", "PAN2DenoiseStep"),
        ("decode", "PAN2VaeDecoderStep"),
    ],
}


class PAN2ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = PAN2ModularPipeline
    pipeline_blocks_class = PAN2Blocks
    pretrained_model_name_or_path = "wuqing157/tiny-pan2-modular-pipe"
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    expected_workflow_blocks = PAN2_WORKFLOWS
    output_name = "videos"

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        pipe = super().get_pipeline(components_manager, dtype)
        pipe.disable_safety_checker()
        return pipe

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "output_type": "pt",
        }
        return inputs


class TestPAN2ModularPipelineFast(PAN2ModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="PAN2 takes a single prompt per call, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="PAN2 takes a single prompt per call, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    def test_num_videos_per_prompt(self):
        pipe = self.get_pipeline().to(torch_device)
        videos = pipe(**self.get_dummy_inputs(), num_videos_per_prompt=2, output=self.output_name)
        assert videos.shape == (2, 5, 3, 32, 32)

    def test_i2v(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = PIL.Image.new("RGB", (40, 48))
        videos = pipe(**inputs, output=self.output_name)
        assert videos.shape == (1, 5, 3, 32, 32)

    def test_prompt_list_raises(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["A cat", "A painting of a squirrel eating a burger"]
        with pytest.raises(ValueError, match="single string"):
            pipe(**inputs, output=self.output_name)

    def test_image_list_raises(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = [PIL.Image.new("RGB", (40, 48))] * 2
        with pytest.raises(ValueError, match="single PIL image"):
            pipe(**inputs, output=self.output_name)

    def test_safety_checker_required_by_default(self):
        pipe = super(PAN2ModularPipelineTesterConfig, self).get_pipeline().to(torch_device)
        assert pipe.config.enable_safety_checker
        with pytest.raises(ValueError, match="requires a safety checker"):
            pipe(**self.get_dummy_inputs(), output=self.output_name)

    def test_latent_output_skips_decoding(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = {**self.get_dummy_inputs(), "output_type": "latent"}
        latents = pipe(**inputs, output=self.output_name)

        num_channels_latents = pipe.transformer.config.out_channels
        assert latents.shape == (1, num_channels_latents, 2, 2, 2)


class TestPAN2ModularPipelineLoading(PAN2ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    def test_save_from_pretrained(self, tmp_path, base_pipe_output):
        base_pipe = self.get_pipeline().to(torch_device)
        base_pipe.save_pretrained(str(tmp_path))

        pipe = ModularPipeline.from_pretrained(tmp_path)
        pipe.load_components(dtype=torch.float32)
        pipe.disable_safety_checker()
        pipe.to(torch_device)

        videos = pipe(**self.get_dummy_inputs(), output=self.output_name)
        assert torch.abs(base_pipe_output - videos).max() < 1e-3


class TestPAN2ModularPipelineWorkflow(PAN2ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestPAN2ModularPipelineMemory(PAN2ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestPAN2ModularPipelineGuider(PAN2ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    pass
