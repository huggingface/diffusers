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

from diffusers.modular_pipelines import ErnieImageAutoBlocks, ErnieImageModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


ERNIE_IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "ErnieImageTextEncoderStep"),
        ("denoise.input", "ErnieImageTextInputStep"),
        ("denoise.set_timesteps", "ErnieImageSetTimestepsStep"),
        ("denoise.prepare_latents", "ErnieImagePrepareLatentsStep"),
        ("denoise.denoise", "ErnieImageDenoiseStep"),
        ("decode", "ErnieImageVaeDecoderStep"),
    ],
}


class TestErnieImageModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = ErnieImageModularPipeline
    pipeline_blocks_class = ErnieImageAutoBlocks
    pretrained_model_name_or_path = "akshan-main/tiny-ernie-image-modular-pipe"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_inference_steps", "num_images_per_prompt", "latents"])
    expected_workflow_blocks = ERNIE_IMAGE_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }

    @pytest.mark.skip(reason="PE generation is non-deterministic on CPU")
    def test_float16_inference(self):
        pass
