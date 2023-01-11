# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, Transformer2DModel

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class DiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DiTPipeline
    test_cpu_offload = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        dit = Transformer2DModel(
            sample_size=4,
            num_layers=2,
            patch_size=2,
            attention_head_dim=2,
            num_attention_heads=2,
            in_channels=4,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            use_ada_layer_norm_zero=True,
            norm_elementwise_affine=False,
        )
        vae = AutoencoderKL()
        scheduler = DDIMScheduler()
        components = {"dit": dit, "vae": vae, "scheduler": scheduler}
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
            "output_type": "numpy",
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
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 4, 4, 3))
        expected_slice = np.array(
            [0.55699474, 0.51614773, 0.6138116, 0.62340677, 0.48881626, 0.41199622, 0.54979277, 0.49198648, 0.5046878]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_consistent(self):
        pass

    def test_inference_batch_single_identical(self):
        pass
