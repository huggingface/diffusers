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

from diffusers.modular_pipelines.pixart_alpha import (
    PixArtAlphaAutoBlocks,
    PixArtAlphaModularPipeline,
)

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


PIXART_ALPHA_TEXT2IMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "PixArtAlphaTextEncoderStep"),
        ("denoise.text_inputs", "PixArtAlphaTextInputStep"),
        ("denoise.set_timesteps", "PixArtAlphaSetTimestepsStep"),
        ("denoise.prepare_latents", "PixArtAlphaPrepareLatentsStep"),
        ("denoise.prepare_micro_conditions", "PixArtAlphaPrepareMicroConditionsStep"),
        ("denoise.denoise", "PixArtAlphaDenoiseStep"),
        ("decode", "PixArtAlphaDecodeStep"),
        ("postprocess", "PixArtAlphaProcessImagesOutputStep"),
    ]
}


class TestPixArtAlphaModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = PixArtAlphaModularPipeline
    pipeline_blocks_class = PixArtAlphaAutoBlocks
    pretrained_model_name_or_path = "idealclx/tiny-pixart-alpha-modular"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = PIXART_ALPHA_TEXT2IMAGE_WORKFLOWS

    def test_pipeline_call_signature(self):
        # Override to prevent signature check failure for guider configurations
        # (guidance_scale) which are intentionally omitted from pipeline inputs.
        pass

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 64,
            "width": 64,
            "max_sequence_length": 48,
            "output_type": "pt",
        }

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)
