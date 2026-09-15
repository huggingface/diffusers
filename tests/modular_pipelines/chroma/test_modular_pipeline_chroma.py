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


from diffusers.modular_pipelines import ChromaAutoBlocks, ChromaModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


CHROMA_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "ChromaTextEncoderStep"),
        ("denoise.input", "ChromaTextInputStep"),
        ("denoise.prepare_latents", "ChromaPrepareLatentsStep"),
        ("denoise.set_timesteps", "ChromaSetTimestepsStep"),
        ("denoise.prepare_attention_mask", "ChromaPrepareAttentionMaskStep"),
        ("denoise.prepare_rope_inputs", "ChromaRoPEInputsStep"),
        ("denoise.denoise", "ChromaDenoiseStep"),
        ("decode", "ChromaDecodeStep"),
    ],
}


class TestChromaModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = ChromaModularPipeline
    pipeline_blocks_class = ChromaAutoBlocks
    pretrained_model_name_or_path = "charchits7/tiny-chroma-modular-pipe"

    params = frozenset(["prompt", "negative_prompt", "height", "width"])
    batch_params = frozenset(["prompt", "negative_prompt"])
    expected_workflow_blocks = CHROMA_WORKFLOWS

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
