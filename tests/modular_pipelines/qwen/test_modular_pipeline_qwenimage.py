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

import unittest

import numpy as np
import PIL
import torch

from diffusers import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    QwenImageAutoBlocks,
    QwenImageEditAutoBlocks,
    QwenImageEditModularPipeline,
    QwenImageModularPipeline,
)

from ...testing_utils import torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class QwenImageModularTests:
    pipeline_class = QwenImageModularPipeline
    pipeline_blocks_class = QwenImageAutoBlocks
    repo = "hf-internal-testing/tiny-qwenimage-modular"

    params = frozenset(["prompt", "height", "width", "negative_prompt", "attention_kwargs", "image", "mask_image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = self.pipeline_blocks_class().init_pipeline(self.repo, components_manager=components_manager)
        pipeline.load_components(torch_dtype=torch_dtype)
        pipeline.set_progress_bar_config(disable=None)
        return pipeline

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs


class QwenImageModularGuiderTests:
    def test_guider_cfg(self):
        pipe = self.get_pipeline()
        pipe = pipe.to(torch_device)

        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs(torch_device)
        out_no_cfg = pipe(**inputs, output="images")

        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self.get_dummy_inputs(torch_device)
        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape
        max_diff = np.abs(out_cfg - out_no_cfg).max()
        assert max_diff > 1e-2, "Output with CFG must be different from normal inference"


class QwenImageModularPipelineFastTests(
    QwenImageModularTests, QwenImageModularGuiderTests, ModularPipelineTesterMixin, unittest.TestCase
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QwenImageEditModularPipelineFastTests(
    QwenImageModularTests, QwenImageModularGuiderTests, ModularPipelineTesterMixin, unittest.TestCase
):
    pipeline_class = QwenImageEditModularPipeline
    pipeline_blocks_class = QwenImageEditAutoBlocks
    repo = "hf-internal-testing/tiny-qwenimage-edit-modular"

    def get_dummy_inputs(self, device, seed=0):
        inputs = super().get_dummy_inputs(device, seed)
        inputs["image"] = PIL.Image.new("RGB", (32, 32), 0)
        return inputs
