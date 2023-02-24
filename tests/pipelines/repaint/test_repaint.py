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

from diffusers import RePaintPipeline, RePaintScheduler, UNet2DModel
from diffusers.utils.testing_utils import load_image, load_numpy, nightly, require_torch_gpu, skip_mps, torch_device

from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class RepaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = RePaintPipeline
    test_cpu_offload = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        torch.manual_seed(0)
        unet = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = RePaintScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        image = np.random.RandomState(seed).standard_normal((1, 3, 32, 32))
        image = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        mask = (image > 0).to(device=device, dtype=torch.float32)
        inputs = {
            "image": image,
            "mask_image": mask,
            "generator": generator,
            "num_inference_steps": 5,
            "eta": 0.0,
            "jump_length": 2,
            "jump_n_sample": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_repaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = RePaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([1.0000, 0.5426, 0.5497, 0.2200, 1.0000, 1.0000, 0.5623, 1.0000, 0.6274])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local()

    # RePaint can hardly be made deterministic since the scheduler is currently always
    # nondeterministic
    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()


@nightly
@require_torch_gpu
class RepaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_celebahq(self):
        original_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/"
            "repaint/celeba_hq_256.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/"
            "repaint/celeba_hq_256_result.npy"
        )

        model_id = "google/ddpm-ema-celebahq-256"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = RePaintScheduler.from_pretrained(model_id)

        repaint = RePaintPipeline(unet=unet, scheduler=scheduler).to(torch_device)
        repaint.set_progress_bar_config(disable=None)
        repaint.enable_attention_slicing()

        generator = torch.manual_seed(0)
        output = repaint(
            original_image,
            mask_image,
            num_inference_steps=250,
            eta=0.0,
            jump_length=10,
            jump_n_sample=10,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).mean() < 1e-2
