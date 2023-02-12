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

import gc
import unittest

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DPMSolverMultistepScheduler, Transformer2DModel
from diffusers.utils import load_numpy, slow
from diffusers.utils.testing_utils import require_torch_gpu

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class DiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DiTPipeline
    test_cpu_offload = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Transformer2DModel(
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
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
        )
        vae = AutoencoderKL()
        scheduler = DDIMScheduler()
        components = {"transformer": transformer.eval(), "vae": vae.eval(), "scheduler": scheduler}
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
            [0.44405967, 0.33592293, 0.6093237, 0.48981372, 0.79098296, 0.7504172, 0.59413105, 0.49462673, 0.35190058]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(relax_max_difference=True)


@require_torch_gpu
@slow
class DiTPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_dit_256(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
        pipe.to("cuda")

        words = ["vase", "umbrella", "white shark", "white wolf"]
        ids = pipe.get_label_ids(words)

        images = pipe(ids, generator=generator, num_inference_steps=40, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}.npy"
            )
            assert np.abs((expected_image - image).sum()) < 1e-3

    def test_dit_512_fp16(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        words = ["vase", "umbrella", "white shark", "white wolf"]
        ids = pipe.get_label_ids(words)

        images = pipe(ids, generator=generator, num_inference_steps=25, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
                f"/dit/{word}_fp16.npy"
            )
            assert np.abs((expected_image - image).sum()) < 1e-3
