# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from collections import OrderedDict

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINTING_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
)
from diffusers.utils import slow


PRETRAINED_MODEL_REPO_MAPPING = OrderedDict(
    [
        ("stable-diffusion", "runwayml/stable-diffusion-v1-5"),
        ("if", "DeepFloyd/IF-I-XL-v1.0"),
        ("kandinsky", "kandinsky-community/kandinsky-2-1"),
        ("kdnsinskyv22", "kandinsky-community/kandinsky-2-2-decoder"),
    ]
)


class AutoPipelineFastTest(unittest.TestCase):
    def test_from_pipe_consistent(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe", requires_safety_checker=False
        )
        original_config = dict(pipe.config)

        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        assert dict(pipe.config) == original_config

        pipe = AutoPipelineForText2Image.from_pipe(pipe)
        assert dict(pipe.config) == original_config

    def test_from_pipe_override(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe", requires_safety_checker=False
        )
        dict(pipe.config)

        pipe = AutoPipelineForImage2Image.from_pipe(pipe, requires_safety_checker=True)
        assert pipe.config.requires_safety_checker is True

        pipe = AutoPipelineForText2Image.from_pipe(pipe, requires_safety_checker=True)
        assert pipe.config.requires_safety_checker is True

    def test_from_pipe_consistent_sdxl(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            requires_aesthetics_score=True,
            force_zeros_for_empty_prompt=False,
        )

        original_config = dict(pipe.config)

        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        assert dict(pipe.config) == original_config

        pipe = AutoPipelineForText2Image.from_pipe(pipe)
        assert dict(pipe.config) == original_config


@slow
class AutoPipelineIntegrationTest(unittest.TestCase):
    def test_pipe_auto(self):
        for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
            # test from_pretrained
            pipe_txt2img = AutoPipelineForText2Image.from_pretrained(model_repo)
            self.assertIsInstance(pipe_txt2img, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_img2img = AutoPipelineForImage2Image.from_pretrained(model_repo)
            self.assertIsInstance(pipe_img2img, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_inpaint = AutoPipelineForInpainting.from_pretrained(model_repo)
            self.assertIsInstance(pipe_inpaint, AUTO_INPAINTING_PIPELINES_MAPPING[model_name])

            # test from_pipe
            for pipe_from in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
                pipe_to = AutoPipelineForText2Image.from_pipe(pipe_from)
                self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

                pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_from)
                self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

                pipe_to = AutoPipelineForInpainting.from_pipe(pipe_from)
                self.assertIsInstance(pipe_to, AUTO_INPAINTING_PIPELINES_MAPPING[model_name])

    def test_from_pipe_consistent(self):
        for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
            # test from_pretrained
            pipe_txt2img = AutoPipelineForText2Image.from_pretrained(model_repo)
            pipe_txt2img_config = dict(pipe_txt2img.config)
            pipe_img2img = AutoPipelineForImage2Image.from_pretrained(model_repo)
            pipe_img2img_config = dict(pipe_txt2img.config)
            pipe_inpaint = AutoPipelineForInpainting.from_pretrained(model_repo)
            pipe_inpaint_config = dict(pipe_txt2img.config)

            # test from_pipe
            for pipe_from in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
                pipe_to = AutoPipelineForText2Image.from_pipe(pipe_from)
                self.assertEqual(dict(pipe_to.config), pipe_txt2img_config)

                pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_from)
                self.assertEqual(dict(pipe_to.config), pipe_img2img_config)

                pipe_to = AutoPipelineForInpainting.from_pipe(pipe_from)
                self.assertEqual(dict(pipe_to.config), pipe_inpaint_config)
