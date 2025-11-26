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


import PIL
import pytest

from diffusers.modular_pipelines import (
    QwenImageAutoBlocks,
    QwenImageEditAutoBlocks,
    QwenImageEditModularPipeline,
    QwenImageEditPlusAutoBlocks,
    QwenImageEditPlusModularPipeline,
    QwenImageModularPipeline,
)

from ..test_modular_pipelines_common import ModularGuiderTesterMixin, ModularPipelineTesterMixin


class TestQwenImageModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageModularPipeline
    pipeline_blocks_class = QwenImageAutoBlocks
    repo = "hf-internal-testing/tiny-qwenimage-modular"

    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image", "mask_image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-4)


class TestQwenImageEditModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageEditModularPipeline
    pipeline_blocks_class = QwenImageEditAutoBlocks
    repo = "hf-internal-testing/tiny-qwenimage-edit-modular"

    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image", "mask_image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        inputs["image"] = PIL.Image.new("RGB", (32, 32), 0)
        return inputs

    def test_guider_cfg(self):
        super().test_guider_cfg(7e-5)


class TestQwenImageEditPlusModularPipelineFast(ModularPipelineTesterMixin, ModularGuiderTesterMixin):
    pipeline_class = QwenImageEditPlusModularPipeline
    pipeline_blocks_class = QwenImageEditPlusAutoBlocks
    repo = "hf-internal-testing/tiny-qwenimage-edit-plus-modular"

    # No `mask_image` yet.
    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image"])

    def get_dummy_inputs(self):
        generator = self.get_generator()
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pt",
        }
        inputs["image"] = PIL.Image.new("RGB", (32, 32), 0)
        return inputs

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_num_images_per_prompt(self):
        super().test_num_images_per_prompt()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_consistent():
        super().test_inference_batch_consistent()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_single_identical():
        super().test_inference_batch_single_identical()

    def test_guider_cfg(self):
        super().test_guider_cfg(1e-3)
