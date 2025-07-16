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
    StableDiffusionXLAutoBlocks,
    StableDiffusionXLModularPipeline,
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
from ..test_modular_pipelines_common import (
    ModularIPAdapterTesterMixin,
    ModularPipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLModularPipelineFastTests(
    ModularIPAdapterTesterMixin,
    ModularPipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLModularPipeline
    pipeline_blocks_class = StableDiffusionXLAutoBlocks
    repo = "hf-internal-testing/tiny-sdxl-modular"
    params = (TEXT_TO_IMAGE_PARAMS | IMAGE_INPAINTING_PARAMS) - {
        "guidance_scale",
        "prompt_embeds",
        "negative_prompt_embeds",
    }
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS | IMAGE_INPAINTING_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = self.pipeline_blocks_class().init_pipeline(self.repo, components_manager=components_manager)
        pipeline.load_default_components(torch_dtype=torch_dtype)
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
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_xl_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = self.get_pipeline()
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.5966781, 0.62939394, 0.48465094, 0.51573336, 0.57593524, 0.47035995, 0.53410417, 0.51436996, 0.47313565]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2, (
            f"image_slice: {image_slice.flatten()}, expected_slice: {expected_slice.flatten()}"
        )

    def test_stable_diffusion_xl_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = self.get_pipeline()
        sd_pipe.update_components(scheduler=LCMScheduler.from_config(sd_pipe.scheduler.config))
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.6880376, 0.6511651, 0.587455, 0.61763, 0.55432945, 0.52064973, 0.5783733, 0.54915607, 0.5460011]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2, (
            f"image_slice: {image_slice.flatten()}, expected_slice: {expected_slice.flatten()}"
        )

    @require_torch_accelerator
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        sd_pipe = self.get_pipeline().to(torch_device)
        pipes.append(sd_pipe)

        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        sd_pipe = self.get_pipeline(components_manager=cm)
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_stable_diffusion_xl_multi_prompts(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs(torch_device)
        output = sd_pipe(**inputs, output="images")
        image_slice_1 = output[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs, output="images")
        image_slice_2 = output[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_3 = output[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_1 = output[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs, output="images")
        image_slice_2 = output[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs, output="images")
        image_slice_3 = output[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        sd_pipe = self.get_pipeline().to(device)
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
        sd_pipe = self.get_pipeline().to(torch_device)
        pipes.append(sd_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd_pipe.save_pretrained(tmpdirname)
            sd_pipe = ModularPipeline.from_pretrained(tmpdirname).to(torch_device)
            sd_pipe.load_default_components(torch_dtype=torch.float32)
            sd_pipe.to(torch_device)
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)
