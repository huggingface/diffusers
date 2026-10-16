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
from PIL import Image

from diffusers import WanVideoToVideoPipeline
from diffusers.modular_pipelines import (
    WanBlocks,
    WanModularPipeline,
    WanVideoToVideoBlocks,
    WanVideoToVideoModularPipeline,
)

from ...testing_utils import assert_tensors_close
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


class WanModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = WanModularPipeline
    pipeline_blocks_class = WanBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-modular-pipe"
    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    output_name = "videos"

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


class TestWanModularPipelineFast(WanModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass


class TestWanModularPipelineLoading(WanModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestWanModularPipelineWorkflow(WanModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestWanModularPipelineMemory(WanModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class WanVideoToVideoModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = WanVideoToVideoModularPipeline
    pipeline_blocks_class = WanVideoToVideoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-modular-pipe"
    params = frozenset(["prompt", "video", "height", "width", "strength"])
    batch_params = frozenset(["prompt", "video"])
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents", "output_type"])
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "",
            "video": [Image.new("RGB", (16, 16))] * 9,
            "generator": self.get_generator(seed),
            "num_inference_steps": 4,
            "height": 16,
            "width": 16,
            "strength": 0.75,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestWanVideoToVideoModularPipelineFast(WanVideoToVideoModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.parametrize(("height", "width"), [(24, 16), (16, 24)])
    def test_height_and_width_must_be_divisible_by_16(self, height, width):
        pipeline = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(height=height, width=width)

        with pytest.raises(ValueError, match="height.*width.*divisible by 16"):
            pipeline(**inputs, output="videos")

    def test_standard_pipeline_parity(self):
        modular_pipeline = self.get_pipeline()
        native_pipeline = WanVideoToVideoPipeline(
            scheduler=modular_pipeline.scheduler,
            text_encoder=modular_pipeline.text_encoder,
            tokenizer=modular_pipeline.tokenizer,
            transformer=modular_pipeline.transformer,
            vae=modular_pipeline.vae,
        )

        modular_output = modular_pipeline(**self.get_dummy_inputs(), output="videos")
        native_inputs = self.get_dummy_inputs()
        native_output = native_pipeline(guidance_scale=5.0, **native_inputs).frames

        assert_tensors_close(modular_output[0], native_output[0], atol=1e-4)


class TestWanVideoToVideoModularPipelineLoading(WanVideoToVideoModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestWanVideoToVideoModularPipelineMemory(WanVideoToVideoModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
