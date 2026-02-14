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


from diffusers.modular_pipelines import ZImageAutoBlocks, ZImageModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


ZIMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("input", "ZImageTextInputStep"),
        ("prepare_latents", "ZImagePrepareLatentsStep"),
        ("set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
    "image2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("vae_encoder", "ZImageVaeImageEncoderStep"),
        ("input", "ZImageTextInputStep"),
        ("additional_inputs", "ZImageAdditionalInputsStep"),
        ("prepare_latents", "ZImagePrepareLatentsStep"),
        ("set_timesteps", "ZImageSetTimestepsStep"),
        ("set_timesteps_with_strength", "ZImageSetTimestepsWithStrengthStep"),
        ("prepare_latents_with_image", "ZImagePrepareLatentswithImageStep"),
        ("denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
}

class TestZImageModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = ZImageModularPipeline
    pipeline_blocks_class = ZImageAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-zimage-modular-pipe"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = ZIMAGE_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-3)
