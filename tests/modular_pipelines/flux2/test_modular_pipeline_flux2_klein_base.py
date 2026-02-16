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

import random

import numpy as np
import PIL
import pytest

from diffusers.modular_pipelines import (
    Flux2KleinBaseAutoBlocks,
    Flux2KleinBaseModularPipeline,
)

from ...testing_utils import floats_tensor, torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


FLUX2_KLEIN_BASE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "Flux2KleinBaseTextEncoderStep"),
        ("denoise.input", "Flux2KleinBaseTextInputStep"),
        ("denoise.prepare_latents", "Flux2PrepareLatentsStep"),
        ("denoise.set_timesteps", "Flux2SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux2KleinBaseRoPEInputsStep"),
        ("denoise.denoise", "Flux2KleinBaseDenoiseStep"),
        ("denoise.after_denoise", "Flux2UnpackLatentsStep"),
        ("decode", "Flux2DecodeStep"),
    ],
}


class TestFlux2KleinBaseModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Flux2KleinBaseModularPipeline
    pipeline_blocks_class = Flux2KleinBaseAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux2-klein-base-modular"

    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = FLUX2_KLEIN_BASE_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            # TODO (Dhruv): Update text encoder config so that vocab_size matches tokenizer
            "max_sequence_length": 8,  # bit of a hack to workaround vocab size mismatch
            "text_encoder_out_layers": (1,),
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        return inputs

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)


FLUX2_KLEIN_BASE_IMAGE_CONDITIONED_WORKFLOWS = {
    "image_conditioned": [
        ("text_encoder", "Flux2KleinBaseTextEncoderStep"),
        ("vae_encoder.preprocess", "Flux2ProcessImagesInputStep"),
        ("vae_encoder.encode", "Flux2VaeEncoderStep"),
        ("denoise.input", "Flux2KleinBaseTextInputStep"),
        ("denoise.prepare_latents", "Flux2PrepareLatentsStep"),
        ("denoise.prepare_image_latents", "Flux2PrepareImageLatentsStep"),
        ("denoise.set_timesteps", "Flux2SetTimestepsStep"),
        ("denoise.prepare_rope_inputs", "Flux2KleinBaseRoPEInputsStep"),
        ("denoise.denoise", "Flux2KleinBaseDenoiseStep"),
        ("denoise.after_denoise", "Flux2UnpackLatentsStep"),
        ("decode", "Flux2DecodeStep"),
    ],
}


class TestFlux2KleinBaseImageConditionedModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Flux2KleinBaseModularPipeline
    pipeline_blocks_class = Flux2KleinBaseAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux2-klein-base-modular"

    params = frozenset(["prompt", "height", "width", "image"])
    batch_params = frozenset(["prompt", "image"])
    expected_workflow_blocks = FLUX2_KLEIN_BASE_IMAGE_CONDITIONED_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            # TODO (Dhruv): Update text encoder config so that vocab_size matches tokenizer
            "max_sequence_length": 8,  # bit of a hack to workaround vocab size mismatch
            "text_encoder_out_layers": (1,),
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = PIL.Image.fromarray(np.uint8(image * 255)).convert("RGB")
        inputs["image"] = init_image

        return inputs

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)

    @pytest.mark.skip(reason="batched inference is currently not supported")
    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=0.0001):
        return
