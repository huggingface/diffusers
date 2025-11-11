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


class TestFluxModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = FluxModularPipeline
    pipeline_blocks_class = FluxAutoBlocks
    repo = "hf-internal-testing/tiny-flux-modular"

    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "pt",
        }
        return inputs

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)


class TestFluxImg2ImgModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = FluxModularPipeline
    pipeline_blocks_class = FluxAutoBlocks
    repo = "hf-internal-testing/tiny-flux-modular"

    params = frozenset(["prompt", "height", "width", "guidance_scale", "image"])
    batch_params = frozenset(["prompt", "image"])

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = super().get_pipeline(components_manager, torch_dtype)

        # Override `vae_scale_factor` here as currently, `image_processor` is initialized with
        # fixed constants instead of
        # https://github.com/huggingface/diffusers/blob/d54622c2679d700b425ad61abce9b80fc36212c0/src/diffusers/pipelines/flux/pipeline_flux_img2img.py#L230C9-L232C10
        pipeline.image_processor = VaeImageProcessor(vae_scale_factor=2)
        return pipeline

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 4,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "pt",
        }
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = PIL.Image.fromarray(np.uint8(image)).convert("RGB")

        inputs["image"] = init_image
        inputs["strength"] = 0.5

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
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_float16_inference(self):
        super().test_float16_inference(8e-2)


class TestFluxKontextModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = FluxKontextModularPipeline
    pipeline_blocks_class = FluxKontextAutoBlocks
    repo = "hf-internal-testing/tiny-flux-kontext-pipe"

    params = frozenset(["prompt", "height", "width", "guidance_scale", "image"])
    batch_params = frozenset(["prompt", "image"])

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "pt",
        }
        image = PIL.Image.new("RGB", (32, 32), 0)

        inputs["image"] = image
        inputs["max_area"] = inputs["height"] * inputs["width"]
        inputs["_auto_resize"] = False

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
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)
