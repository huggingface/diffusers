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

import random
import tempfile
import unittest

import numpy as np
import PIL
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import (
    FluxAutoBlocks,
    FluxKontextAutoBlocks,
    FluxKontextModularPipeline,
    FluxModularPipeline,
    ModularPipeline,
)

from ...testing_utils import floats_tensor, torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class FluxModularTests:
    pipeline_class = FluxModularPipeline
    pipeline_blocks_class = FluxAutoBlocks
    repo = "hf-internal-testing/tiny-flux-modular"

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


class FluxModularPipelineFastTests(FluxModularTests, ModularPipelineTesterMixin, unittest.TestCase):
    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])


class FluxImg2ImgModularPipelineFastTests(FluxModularTests, ModularPipelineTesterMixin, unittest.TestCase):
    params = frozenset(["prompt", "height", "width", "guidance_scale", "image"])
    batch_params = frozenset(["prompt", "image"])

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = super().get_pipeline(components_manager, torch_dtype)
        # Override `vae_scale_factor` here as currently, `image_processor` is initialized with
        # fixed constants instead of
        # https://github.com/huggingface/diffusers/blob/d54622c2679d700b425ad61abce9b80fc36212c0/src/diffusers/pipelines/flux/pipeline_flux_img2img.py#L230C9-L232C10
        pipeline.image_processor = VaeImageProcessor(vae_scale_factor=2)
        return pipeline

    def get_dummy_inputs(self, device, seed=0):
        inputs = super().get_dummy_inputs(device, seed)
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        inputs["image"] = image
        inputs["strength"] = 0.8
        inputs["height"] = 8
        inputs["width"] = 8
        return inputs

    def test_save_from_pretrained(self):
        pipes = []
        base_pipe = self.get_pipeline().to(torch_device)
        pipes.append(base_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            base_pipe.save_pretrained(tmpdirname)
            pipe = ModularPipeline.from_pretrained(tmpdirname).to(torch_device)
            pipe.load_components(torch_dtype=torch.float32)
            pipe.to(torch_device)
            pipe.image_processor = VaeImageProcessor(vae_scale_factor=2)

        pipes.append(pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3


class FluxKontextModularPipelineFastTests(FluxImg2ImgModularPipelineFastTests):
    pipeline_class = FluxKontextModularPipeline
    pipeline_blocks_class = FluxKontextAutoBlocks
    repo = "hf-internal-testing/tiny-flux-kontext-pipe"

    def get_dummy_inputs(self, device, seed=0):
        inputs = super().get_dummy_inputs(device, seed)
        image = PIL.Image.new("RGB", (32, 32), 0)
        _ = inputs.pop("strength")
        inputs["image"] = image
        inputs["height"] = 8
        inputs["width"] = 8
        inputs["max_area"] = 8 * 8
        inputs["_auto_resize"] = False
        return inputs
