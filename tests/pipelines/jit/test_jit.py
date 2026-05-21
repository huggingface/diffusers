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

import unittest

import numpy as np
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, JiTPipeline
from diffusers.models.transformers import JiTTransformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
)

from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class JiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = JiTPipeline
    params = CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = JiTTransformer2DModel(
            sample_size=32,
            patch_size=4,
            in_channels=3,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            mlp_ratio=4.0,
            num_classes=10,
            bottleneck_dim=16,
            in_context_len=4,
            in_context_start=1,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        components = {"transformer": transformer.eval(), "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "class_labels": [1],
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images

        self.assertEqual(image.shape, (1, 32, 32, 3))
        self.assertTrue(np.all(image >= 0.0) and np.all(image <= 1.0))

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_model_cpu_offload_forward_pass(self):
        self.skipTest("Single-model pipeline keeps model on device with enable_model_cpu_offload.")

    def test_cpu_offload_forward_pass_twice(self):
        self.skipTest("Single-model pipeline keeps model on device with enable_model_cpu_offload.")


