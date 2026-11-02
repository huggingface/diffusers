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

import json

import PIL.Image
import pytest
import torch

from diffusers.modular_pipelines import PAN2Blocks, PAN2ModularPipeline
from diffusers.modular_pipelines.pan2.encoders import (
    PAN2_DEFAULT_NEGATIVE_PROMPT,
    get_pan2_default_negative_prompt,
)

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
        ("denoise.image_input", "PAN2Image2VideoInputStep"),
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
    pretrained_model_name_or_path = "hf-internal-testing/tiny-pan2-modular-pipe"
    expected_latents_shape = (1, 48, 2, 2, 2)
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    expected_workflow_blocks = PAN2_WORKFLOWS
    output_name = "videos"

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        pipe = super().get_pipeline(components_manager, dtype)
        # Disable the checks in the config, so that a pipeline saved and reloaded by the tests keeps them off.
        pipe.register_to_config(enable_safety_checker=False)
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
    @pytest.mark.skip(reason="PAN 2 takes a single prompt per call, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="PAN 2 takes a single prompt per call, so a batch of prompts is not a thing.")
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

    def test_i2v_takes_size_from_image_latents(self):
        pipe = self.get_pipeline().to(torch_device)
        vae_encoder = pipe.blocks.sub_blocks["vae_encoder"].init_pipeline()
        vae_encoder.update_components(vae=pipe.vae, video_processor=pipe.video_processor)
        image_latents = vae_encoder(image=PIL.Image.new("RGB", (40, 48)), height=32, width=64, output="image_latents")

        inputs = {k: v for k, v in self.get_dummy_inputs().items() if k not in ("height", "width")}
        videos = pipe(**inputs, image_latents=image_latents, output=self.output_name)
        assert videos.shape == (1, 5, 3, 32, 64)
        with pytest.raises(ValueError, match="do not match the size of `image_latents`"):
            pipe(**inputs, image_latents=image_latents, height=64, output=self.output_name)

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

    def test_default_negative_prompt_spans_the_clip(self):
        for num_frames, fps, span in [(241, 24, "0 to 10"), (313, 24, "0 to 13"), (85, 24, "0 to 3.5")]:
            negative = json.loads(get_pan2_default_negative_prompt(num_frames, fps))
            event = negative["events"][0]
            ranges = [event["time_range_sec"], event["camera_motion"][0]["time_range_sec"]]
            ranges.append(event["contained_actions"][0]["time_range_sec"])
            assert ranges == [span] * 3
        assert json.loads(get_pan2_default_negative_prompt(241, 24)) == PAN2_DEFAULT_NEGATIVE_PROMPT

    def test_missing_negative_prompt_uses_the_default(self):
        pipe = self.get_pipeline().to(torch_device)
        default = pipe(**self.get_dummy_inputs(), output=self.output_name)
        empty = pipe(**self.get_dummy_inputs(), negative_prompt="", output=self.output_name)
        rendered = pipe(
            **self.get_dummy_inputs(),
            negative_prompt=get_pan2_default_negative_prompt(5, 24),
            output=self.output_name,
        )
        other = pipe(**self.get_dummy_inputs(), negative_prompt="blurry", output=self.output_name)
        assert torch.equal(default, empty)
        assert torch.equal(default, rendered)
        assert not torch.equal(default, other)

    def test_safety_checker_required_by_default(self):
        pipe = super(PAN2ModularPipelineTesterConfig, self).get_pipeline()
        assert pipe.config.enable_safety_checker
        assert pipe.requires_safety_checker


class TestPAN2ModularPipelineLoading(PAN2ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestPAN2ModularPipelineWorkflow(PAN2ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestPAN2ModularPipelineMemory(PAN2ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestPAN2ModularPipelineGuider(PAN2ModularPipelineTesterConfig, ModularGuiderTesterMixin):
    pass
