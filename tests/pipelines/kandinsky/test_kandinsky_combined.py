# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from diffusers import KandinskyCombinedPipeline, KandinskyImg2ImgCombinedPipeline, KandinskyInpaintCombinedPipeline
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, torch_device

from ..test_pipelines_common import PipelineTesterMixin
from .test_kandinsky import Dummies
from .test_kandinsky_img2img import Dummies as Img2ImgDummies
from .test_kandinsky_inpaint import Dummies as InpaintDummies
from .test_kandinsky_prior import Dummies as PriorDummies


enable_full_determinism()


class KandinskyPipelineCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyCombinedPipeline
    params = [
        "prompt",
    ]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = True

    def get_dummy_components(self):
        dummy = Dummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()

        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, device, seed=0):
        prior_dummy = PriorDummies()
        inputs = prior_dummy.get_dummy_inputs(device=device, seed=seed)
        inputs.update(
            {
                "height": 64,
                "width": 64,
            }
        )
        return inputs

    def test_kandinsky(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.2893, 0.1464, 0.4603, 0.3529, 0.4612, 0.7701, 0.4027, 0.3051, 0.5155])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_torch_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-2)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=2e-1)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=5e-4)


class KandinskyPipelineImg2ImgCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyImg2ImgCombinedPipeline
    params = ["prompt", "image"]
    batch_params = ["prompt", "negative_prompt", "image"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    def get_dummy_components(self):
        dummy = Img2ImgDummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()

        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, device, seed=0):
        prior_dummy = PriorDummies()
        dummy = Img2ImgDummies()
        inputs = prior_dummy.get_dummy_inputs(device=device, seed=seed)
        inputs.update(dummy.get_dummy_inputs(device=device, seed=seed))
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs

    def test_kandinsky(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.4852, 0.4136, 0.4539, 0.4781, 0.4680, 0.5217, 0.4973, 0.4089, 0.4977])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_torch_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-2)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=5e-4)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=5e-4)


class KandinskyPipelineInpaintCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyInpaintCombinedPipeline
    params = ["prompt", "image", "mask_image"]
    batch_params = ["prompt", "negative_prompt", "image", "mask_image"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    def get_dummy_components(self):
        dummy = InpaintDummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()

        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, device, seed=0):
        prior_dummy = PriorDummies()
        dummy = InpaintDummies()
        inputs = prior_dummy.get_dummy_inputs(device=device, seed=seed)
        inputs.update(dummy.get_dummy_inputs(device=device, seed=seed))
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs

    def test_kandinsky(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]

        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        print(image_from_tuple_slice)

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.0320, 0.0860, 0.4013, 0.0518, 0.2484, 0.5847, 0.4411, 0.2321, 0.4593])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_torch_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-2)

    @unittest.skip("Difference between FP16 and FP32 too large on CI")
    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=5e-4)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=5e-4)

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=5e-3)
