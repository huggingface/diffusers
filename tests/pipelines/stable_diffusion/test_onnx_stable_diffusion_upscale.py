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

import random
import unittest

import numpy as np
import torch

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    OnnxStableDiffusionUpscalePipeline,
    PNDMScheduler,
)
from diffusers.utils import floats_tensor
from diffusers.utils.testing_utils import (
    is_onnx_available,
    load_image,
    nightly,
    require_onnxruntime,
    require_torch_gpu,
)

from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionUpscalePipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # TODO: is there an appropriate internal test set?
    hub_checkpoint = "ssube/stable-diffusion-x4-upscaler-onnx"

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 128, 128), rng=random.Random(seed))
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_pipeline_default_ddpm(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        # started as 128, should now be 512
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.6974782, 0.68902093, 0.70135885, 0.7583618, 0.7804545, 0.7854912, 0.78667426, 0.78743863, 0.78070223]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-1

    def test_pipeline_pndm(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config, skip_prk_steps=True)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.6898892, 0.59240556, 0.52499527, 0.58866215, 0.52258235, 0.52572715, 0.62414473, 0.6174387, 0.6214964]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_dpm_multistep(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.7659278, 0.76437664, 0.75579107, 0.7691116, 0.77666986, 0.7727672, 0.7758664, 0.7812226, 0.76942515]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.6974782, 0.68902093, 0.70135885, 0.7583618, 0.7804545, 0.7854912, 0.78667426, 0.78743863, 0.78070223]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler_ancestral(self):
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.77424496, 0.773601, 0.7645288, 0.7769598, 0.7772739, 0.7738688, 0.78187233, 0.77879584, 0.767043]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1


@nightly
@require_onnxruntime
@require_torch_gpu
class OnnxStableDiffusionUpscalePipelineIntegrationTests(unittest.TestCase):
    @property
    def gpu_provider(self):
        return (
            "CUDAExecutionProvider",
            {
                "gpu_mem_limit": "15000000000",  # 15GB
                "arena_extend_strategy": "kSameAsRequested",
            },
        )

    @property
    def gpu_options(self):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        return options

    def test_inference_default_ddpm(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((128, 128))
        # using the PNDM scheduler by default
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            "ssube/stable-diffusion-x4-upscaler-onnx",
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            guidance_scale=7.5,
            num_inference_steps=10,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4883, 0.4947, 0.4980, 0.4975, 0.4982, 0.4980, 0.5000, 0.5006, 0.4972])
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues

        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2

    def test_inference_k_lms(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((128, 128))
        lms_scheduler = LMSDiscreteScheduler.from_pretrained(
            "ssube/stable-diffusion-x4-upscaler-onnx", subfolder="scheduler"
        )
        pipe = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            "ssube/stable-diffusion-x4-upscaler-onnx",
            scheduler=lms_scheduler,
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.50173753, 0.50223356, 0.502039, 0.50233036, 0.5023725, 0.5022601, 0.5018758, 0.50234085, 0.50241566]
        )
        # TODO: lower the tolerance after finding the cause of onnxruntime reproducibility issues

        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
