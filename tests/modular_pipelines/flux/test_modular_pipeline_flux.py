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

import torch

from diffusers.modular_pipelines import FluxAutoBlocks, FluxModularPipeline

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class FluxModularFastTests(ModularPipelineTesterMixin, unittest.TestCase):
    pipeline_class = FluxModularPipeline
    pipeline_blocks_class = FluxAutoBlocks
    repo = "hf-internal-testing/tiny-flux-modular"
    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = self.pipeline_blocks_class().init_pipeline(self.repo, components_manager=components_manager)
        pipeline.load_components(torch_dtype=torch_dtype)
        return pipeline

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs
