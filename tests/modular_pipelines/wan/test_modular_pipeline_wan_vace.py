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
from PIL import Image

from diffusers.modular_pipelines import Wan22VaceBlocks, Wan22VaceModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class TestWan22VaceModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Wan22VaceModularPipeline
    pipeline_blocks_class = Wan22VaceBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-wan22-vace-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames", "video", "mask", "reference_images"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        num_frames = 9
        video = [Image.new("RGB", (16, 16))] * num_frames
        mask = [Image.new("L", (16, 16), 0)] * num_frames
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "video": video,
            "mask": mask,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": num_frames,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_with_reference_image(self):
        pipe = self.get_pipeline().to("cpu")

        inputs = self.get_dummy_inputs()
        inputs["reference_images"] = Image.new("RGB", (16, 16))
        videos = pipe(**inputs, output=self.output_name)
        assert videos.shape == (1, 9, 3, 16, 16)
        assert torch.isnan(videos).sum() == 0

    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="num_videos_per_prompt")
    def test_num_images_per_prompt(self):
        pass
