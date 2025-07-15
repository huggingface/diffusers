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

import tempfile
import unittest

import numpy as np
import torch

from diffusers import (
    ComponentsManager,
    LCMScheduler,
    ModularPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_accelerator,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_INPAINTING_BATCH_PARAMS,
    IMAGE_INPAINTING_PARAMS,
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLModularPipelineFastTests(
    SDFunctionTesterMixin,
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLPipeline
    params = (TEXT_TO_IMAGE_PARAMS | IMAGE_INPAINTING_PARAMS) - {"guidance_scale"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS | IMAGE_INPAINTING_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    test_layerwise_casting = False
    test_group_offloading = False

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_xl_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe")
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5388, 0.5452, 0.4694, 0.4583, 0.5253, 0.4832, 0.5288, 0.5035, 0.47])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe")
        sd_pipe.update_components(scheduler=LCMScheduler.from_config(sd_pipe.scheduler.config))
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_euler_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe")
        sd_pipe.update_components(scheduler=LCMScheduler.from_config(sd_pipe.scheduler.config))
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @require_torch_accelerator
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        sd_pipe = ModularPipeline.from_pretrained(
            "hf-internal-testing/tiny-sd-pipe",
        ).to(torch_device)
        pipes.append(sd_pipe)

        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe", components_manager=cm).to(
            torch_device
        )
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_stable_diffusion_xl_multi_prompts(self):
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe").to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs(torch_device)
        output = sd_pipe(**inputs, output="images")
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs, output="images")
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs, output="images")
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe").to(torch_device)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, output="images")
        image_slice_with_no_neg_cond = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
            output="images",
        )
        image_slice_with_neg_cond = image[0, -3:, -3:, -1]

        self.assertTrue(np.abs(image_slice_with_no_neg_cond - image_slice_with_neg_cond).max() > 1e-2)

    def test_stable_diffusion_xl_save_from_pretrained(self):
        pipes = []
        sd_pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe").to(torch_device)
        pipes.append(sd_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd_pipe.save_pretrained(tmpdirname)
            sd_pipe = ModularPipeline.from_pretrained(tmpdirname).to(torch_device)
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
