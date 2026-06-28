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


from diffusers.modular_pipelines import Krea2AutoBlocks, Krea2ModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


KREA2_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "Krea2TextEncoderStep"),
        ("denoise.input", "Krea2TextInputsStep"),
        ("denoise.prepare_latents", "Krea2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Krea2SetTimestepsStep"),
        ("denoise.prepare_position_ids", "Krea2PreparePositionIdsStep"),
        ("denoise.denoise", "Krea2DenoiseStep"),
        ("decode", "Krea2DecodeStep"),
    ],
}


class TestKrea2ModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Krea2ModularPipeline
    pipeline_blocks_class = Krea2AutoBlocks
    pretrained_model_name_or_path = "CedricPerauer/tiny-krea2-modular-pipe"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = KREA2_WORKFLOWS

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
