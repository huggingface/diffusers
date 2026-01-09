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

import gc
import os
import shutil
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    DiffusionPipeline,
)
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
)

from ..testing_utils import slow


PRETRAINED_MODEL_REPO_MAPPING = OrderedDict(
    [
        ("stable-diffusion", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        ("if", "DeepFloyd/IF-I-XL-v1.0"),
        ("kandinsky", "kandinsky-community/kandinsky-2-1"),
        ("kandinsky22", "kandinsky-community/kandinsky-2-2-decoder"),
    ]
)


class AutoPipelineFastTest(unittest.TestCase):
    @property
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=1,
            projection_dim=1,
            num_hidden_layers=1,
            num_attention_heads=1,
            image_size=1,
            intermediate_size=1,
            patch_size=1,
        )
        return CLIPVisionModelWithProjection(config)

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

        pipe = AutoPipelineForText2Image.from_pipe(pipe)
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

        assert dict(pipe.config) == original_config

    def test_kwargs_local_files_only(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-torch"
        tmpdirname = DiffusionPipeline.download(repo)
        tmpdirname = Path(tmpdirname)

        # edit commit_id to so that it's not the latest commit
        commit_id = tmpdirname.name
        new_commit_id = commit_id + "hug"

        ref_dir = tmpdirname.parent.parent / "refs/main"
        with open(ref_dir, "w") as f:
            f.write(new_commit_id)

        new_tmpdirname = tmpdirname.parent / new_commit_id
        os.rename(tmpdirname, new_tmpdirname)

        try:
            AutoPipelineForText2Image.from_pretrained(repo, local_files_only=True)
        except OSError:
            assert False, "not able to load local files"

        shutil.rmtree(tmpdirname.parent.parent)

    def test_from_pretrained_text2img(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        pipe = AutoPipelineForText2Image.from_pretrained(repo)
        assert pipe.__class__.__name__ == "StableDiffusionXLPipeline"

        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")
        pipe_control = AutoPipelineForText2Image.from_pretrained(repo, controlnet=controlnet)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"

        pipe_pag = AutoPipelineForText2Image.from_pretrained(repo, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGPipeline"

        pipe_control_pag = AutoPipelineForText2Image.from_pretrained(repo, controlnet=controlnet, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"

    def test_from_pipe_pag_text2img(self):
        # test from StableDiffusionXLPipeline
        pipe = AutoPipelineForText2Image.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")

        #  - test `enable_pag` flag
        pipe_pag = AutoPipelineForText2Image.from_pipe(pipe, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        assert "controlnet" not in pipe_pag.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLPipeline"
        assert "controlnet" not in pipe.components

        #  - test `enabe_pag` + `controlnet` flag
        pipe_control_pag = AutoPipelineForText2Image.from_pipe(pipe, controlnet=controlnet, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"
        assert "controlnet" in pipe_control_pag.components

        pipe_control = AutoPipelineForText2Image.from_pipe(pipe, controlnet=controlnet, enable_pag=False)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe_control.components

        pipe_pag = AutoPipelineForText2Image.from_pipe(pipe, controlnet=None, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        assert "controlnet" not in pipe_pag.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe, controlnet=None, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLPipeline"
        assert "controlnet" not in pipe.components

        # test from StableDiffusionXLControlNetPipeline
        # - test `enable_pag` flag
        pipe_control_pag = AutoPipelineForText2Image.from_pipe(pipe_control, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"
        assert "controlnet" in pipe_control_pag.components

        pipe_control = AutoPipelineForText2Image.from_pipe(pipe_control, enable_pag=False)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe_control.components

        # - test `enable_pag` + `controlnet` flag
        pipe_control_pag = AutoPipelineForText2Image.from_pipe(pipe_control, controlnet=controlnet, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"
        assert "controlnet" in pipe_control_pag.components

        pipe_control = AutoPipelineForText2Image.from_pipe(pipe_control, controlnet=controlnet, enable_pag=False)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe_control.components

        pipe_pag = AutoPipelineForText2Image.from_pipe(pipe_control, controlnet=None, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        assert "controlnet" not in pipe_pag.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe_control, controlnet=None, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLPipeline"
        assert "controlnet" not in pipe.components

        # test from StableDiffusionXLControlNetPAGPipeline
        # - test `enable_pag` flag
        pipe_control_pag = AutoPipelineForText2Image.from_pipe(pipe_control_pag, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"
        assert "controlnet" in pipe_control_pag.components

        pipe_control = AutoPipelineForText2Image.from_pipe(pipe_control_pag, enable_pag=False)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe_control.components

        # - test `enable_pag` + `controlnet` flag
        pipe_control_pag = AutoPipelineForText2Image.from_pipe(
            pipe_control_pag, controlnet=controlnet, enable_pag=True
        )
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGPipeline"
        assert "controlnet" in pipe_control_pag.components

        pipe_control = AutoPipelineForText2Image.from_pipe(pipe_control_pag, controlnet=controlnet, enable_pag=False)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe_control.components

        pipe_pag = AutoPipelineForText2Image.from_pipe(pipe_control_pag, controlnet=None, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        assert "controlnet" not in pipe_pag.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe_control_pag, controlnet=None, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLPipeline"
        assert "controlnet" not in pipe.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe_control_pag, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLControlNetPipeline"
        assert "controlnet" in pipe.components

    def test_from_pretrained_img2img(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        pipe = AutoPipelineForImage2Image.from_pretrained(repo)
        assert pipe.__class__.__name__ == "StableDiffusionXLImg2ImgPipeline"

        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")
        pipe_control = AutoPipelineForImage2Image.from_pretrained(repo, controlnet=controlnet)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetImg2ImgPipeline"

        pipe_pag = AutoPipelineForImage2Image.from_pretrained(repo, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        pipe_control_pag = AutoPipelineForImage2Image.from_pretrained(repo, controlnet=controlnet, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGImg2ImgPipeline"

    def test_from_pretrained_img2img_refiner(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-xl-refiner-pipe"

        pipe = AutoPipelineForImage2Image.from_pretrained(repo)
        assert pipe.__class__.__name__ == "StableDiffusionXLImg2ImgPipeline"

        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")
        pipe_control = AutoPipelineForImage2Image.from_pretrained(repo, controlnet=controlnet)
        assert pipe_control.__class__.__name__ == "StableDiffusionXLControlNetImg2ImgPipeline"

        pipe_pag = AutoPipelineForImage2Image.from_pretrained(repo, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        pipe_control_pag = AutoPipelineForImage2Image.from_pretrained(repo, controlnet=controlnet, enable_pag=True)
        assert pipe_control_pag.__class__.__name__ == "StableDiffusionXLControlNetPAGImg2ImgPipeline"

    def test_from_pipe_pag_img2img(self):
        # test from tableDiffusionXLPAGImg2ImgPipeline
        pipe = AutoPipelineForImage2Image.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        # - test `enable_pag` flag
        pipe_pag = AutoPipelineForImage2Image.from_pipe(pipe, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        pipe = AutoPipelineForImage2Image.from_pipe(pipe, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLImg2ImgPipeline"

        # testing from StableDiffusionXLPAGImg2ImgPipeline
        # - test `enable_pag` flag
        pipe_pag = AutoPipelineForImage2Image.from_pipe(pipe_pag, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        pipe = AutoPipelineForImage2Image.from_pipe(pipe_pag, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLImg2ImgPipeline"

    def test_from_pretrained_inpaint(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"

        pipe = AutoPipelineForInpainting.from_pretrained(repo)
        assert pipe.__class__.__name__ == "StableDiffusionXLInpaintPipeline"

        pipe_pag = AutoPipelineForInpainting.from_pretrained(repo, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"

    def test_from_pretrained_inpaint_from_inpaint(self):
        repo = "hf-internal-testing/tiny-stable-diffusion-xl-inpaint-pipe"

        pipe = AutoPipelineForInpainting.from_pretrained(repo)
        assert pipe.__class__.__name__ == "StableDiffusionXLInpaintPipeline"

        # make sure you can use pag with inpaint-specific pipeline
        pipe = AutoPipelineForInpainting.from_pretrained(repo, enable_pag=True)
        assert pipe.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"

    def test_from_pipe_pag_inpaint(self):
        # test from tableDiffusionXLPAGInpaintPipeline
        pipe = AutoPipelineForInpainting.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        # - test `enable_pag` flag
        pipe_pag = AutoPipelineForInpainting.from_pipe(pipe, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"

        pipe = AutoPipelineForInpainting.from_pipe(pipe, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLInpaintPipeline"

        # testing from StableDiffusionXLPAGInpaintPipeline
        # - test `enable_pag` flag
        pipe_pag = AutoPipelineForInpainting.from_pipe(pipe_pag, enable_pag=True)
        assert pipe_pag.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"

        pipe = AutoPipelineForInpainting.from_pipe(pipe_pag, enable_pag=False)
        assert pipe.__class__.__name__ == "StableDiffusionXLInpaintPipeline"

    def test_from_pipe_pag_new_task(self):
        # for from_pipe_new_task we only need to make sure it can map to the same pipeline from a different task,
        # i.e. no need to test `enable_pag` + `controlnet` flag because it is already tested in `test_from_pipe_pag_text2img` and `test_from_pipe_pag_inpaint`etc
        pipe_pag_text2img = AutoPipelineForText2Image.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe", enable_pag=True
        )

        # text2img pag -> inpaint pag
        pipe_pag_inpaint = AutoPipelineForInpainting.from_pipe(pipe_pag_text2img)
        assert pipe_pag_inpaint.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"
        # text2img pag -> img2img pag
        pipe_pag_img2img = AutoPipelineForImage2Image.from_pipe(pipe_pag_text2img)
        assert pipe_pag_img2img.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        # inpaint pag -> text2img pag
        pipe_pag_text2img = AutoPipelineForText2Image.from_pipe(pipe_pag_inpaint)
        assert pipe_pag_text2img.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        # inpaint pag -> img2img pag
        pipe_pag_img2img = AutoPipelineForImage2Image.from_pipe(pipe_pag_inpaint)
        assert pipe_pag_img2img.__class__.__name__ == "StableDiffusionXLPAGImg2ImgPipeline"

        # img2img pag -> text2img pag
        pipe_pag_text2img = AutoPipelineForText2Image.from_pipe(pipe_pag_img2img)
        assert pipe_pag_text2img.__class__.__name__ == "StableDiffusionXLPAGPipeline"
        # img2img pag -> inpaint pag
        pipe_pag_inpaint = AutoPipelineForInpainting.from_pipe(pipe_pag_img2img)
        assert pipe_pag_inpaint.__class__.__name__ == "StableDiffusionXLPAGInpaintPipeline"

    def test_from_pipe_controlnet_text2img(self):
        pipe = AutoPipelineForText2Image.from_pretrained("hf-internal-testing/tiny-stable-diffusion-pipe")
        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")

        pipe = AutoPipelineForText2Image.from_pipe(pipe, controlnet=controlnet)
        assert pipe.__class__.__name__ == "StableDiffusionControlNetPipeline"
        assert "controlnet" in pipe.components

        pipe = AutoPipelineForText2Image.from_pipe(pipe, controlnet=None)
        assert pipe.__class__.__name__ == "StableDiffusionPipeline"
        assert "controlnet" not in pipe.components

    def test_from_pipe_controlnet_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained("hf-internal-testing/tiny-stable-diffusion-pipe")
        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")

        pipe = AutoPipelineForImage2Image.from_pipe(pipe, controlnet=controlnet)
        assert pipe.__class__.__name__ == "StableDiffusionControlNetImg2ImgPipeline"
        assert "controlnet" in pipe.components

        pipe = AutoPipelineForImage2Image.from_pipe(pipe, controlnet=None)
        assert pipe.__class__.__name__ == "StableDiffusionImg2ImgPipeline"
        assert "controlnet" not in pipe.components

    def test_from_pipe_controlnet_inpaint(self):
        pipe = AutoPipelineForInpainting.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")

        pipe = AutoPipelineForInpainting.from_pipe(pipe, controlnet=controlnet)
        assert pipe.__class__.__name__ == "StableDiffusionControlNetInpaintPipeline"
        assert "controlnet" in pipe.components

        pipe = AutoPipelineForInpainting.from_pipe(pipe, controlnet=None)
        assert pipe.__class__.__name__ == "StableDiffusionInpaintPipeline"
        assert "controlnet" not in pipe.components

    def test_from_pipe_controlnet_new_task(self):
        pipe_text2img = AutoPipelineForText2Image.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        controlnet = ControlNetModel.from_pretrained("hf-internal-testing/tiny-controlnet")

        pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(pipe_text2img, controlnet=controlnet)
        assert pipe_control_img2img.__class__.__name__ == "StableDiffusionControlNetImg2ImgPipeline"
        assert "controlnet" in pipe_control_img2img.components

        pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_control_img2img, controlnet=None)
        assert pipe_inpaint.__class__.__name__ == "StableDiffusionInpaintPipeline"
        assert "controlnet" not in pipe_inpaint.components

        # testing `from_pipe` for text2img controlnet
        ## 1. from a different controlnet pipe, without controlnet argument
        pipe_control_text2img = AutoPipelineForText2Image.from_pipe(pipe_control_img2img)
        assert pipe_control_text2img.__class__.__name__ == "StableDiffusionControlNetPipeline"
        assert "controlnet" in pipe_control_text2img.components

        ## 2. from a different controlnet pipe, with controlnet argument
        pipe_control_text2img = AutoPipelineForText2Image.from_pipe(pipe_control_img2img, controlnet=controlnet)
        assert pipe_control_text2img.__class__.__name__ == "StableDiffusionControlNetPipeline"
        assert "controlnet" in pipe_control_text2img.components

        ## 3. from same controlnet pipeline class, with a different controlnet component
        pipe_control_text2img = AutoPipelineForText2Image.from_pipe(pipe_control_text2img, controlnet=controlnet)
        assert pipe_control_text2img.__class__.__name__ == "StableDiffusionControlNetPipeline"
        assert "controlnet" in pipe_control_text2img.components

        # testing from_pipe for inpainting
        ## 1. from a different controlnet pipeline class
        pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(pipe_control_img2img)
        assert pipe_control_inpaint.__class__.__name__ == "StableDiffusionControlNetInpaintPipeline"
        assert "controlnet" in pipe_control_inpaint.components

        ## from a different controlnet pipe, with a different controlnet
        pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(pipe_control_img2img, controlnet=controlnet)
        assert pipe_control_inpaint.__class__.__name__ == "StableDiffusionControlNetInpaintPipeline"
        assert "controlnet" in pipe_control_inpaint.components

        ## from same controlnet pipe, with a different controlnet
        pipe_control_inpaint = AutoPipelineForInpainting.from_pipe(pipe_control_inpaint, controlnet=controlnet)
        assert pipe_control_inpaint.__class__.__name__ == "StableDiffusionControlNetInpaintPipeline"
        assert "controlnet" in pipe_control_inpaint.components

        # testing from_pipe from img2img controlnet
        ## from a different controlnet pipe, without controlnet argument
        pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(pipe_control_text2img)
        assert pipe_control_img2img.__class__.__name__ == "StableDiffusionControlNetImg2ImgPipeline"
        assert "controlnet" in pipe_control_img2img.components

        # from a different controlnet pipe, with a different controlnet component
        pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(pipe_control_text2img, controlnet=controlnet)
        assert pipe_control_img2img.__class__.__name__ == "StableDiffusionControlNetImg2ImgPipeline"
        assert "controlnet" in pipe_control_img2img.components

        # from same controlnet pipeline class, with a different controlnet
        pipe_control_img2img = AutoPipelineForImage2Image.from_pipe(pipe_control_img2img, controlnet=controlnet)
        assert pipe_control_img2img.__class__.__name__ == "StableDiffusionControlNetImg2ImgPipeline"
        assert "controlnet" in pipe_control_img2img.components

    def test_from_pipe_optional_components(self):
        image_encoder = self.dummy_image_encoder

        pipe = AutoPipelineForText2Image.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            image_encoder=image_encoder,
        )

        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        assert pipe.image_encoder is not None

        pipe = AutoPipelineForText2Image.from_pipe(pipe, image_encoder=None)
        assert pipe.image_encoder is None


@slow
class AutoPipelineIntegrationTest(unittest.TestCase):
    def test_pipe_auto(self):
        for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
            # test txt2img
            pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
                model_repo, variant="fp16", torch_dtype=torch.float16
            )
            self.assertIsInstance(pipe_txt2img, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_to = AutoPipelineForText2Image.from_pipe(pipe_txt2img)
            self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_txt2img)
            self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

            if "kandinsky" not in model_name:
                pipe_to = AutoPipelineForInpainting.from_pipe(pipe_txt2img)
                self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[model_name])

            del pipe_txt2img, pipe_to
            gc.collect()

            # test img2img

            pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
                model_repo, variant="fp16", torch_dtype=torch.float16
            )
            self.assertIsInstance(pipe_img2img, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_to = AutoPipelineForText2Image.from_pipe(pipe_img2img)
            self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

            pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_img2img)
            self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

            if "kandinsky" not in model_name:
                pipe_to = AutoPipelineForInpainting.from_pipe(pipe_img2img)
                self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[model_name])

            del pipe_img2img, pipe_to
            gc.collect()

            # test inpaint

            if "kandinsky" not in model_name:
                pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
                    model_repo, variant="fp16", torch_dtype=torch.float16
                )
                self.assertIsInstance(pipe_inpaint, AUTO_INPAINT_PIPELINES_MAPPING[model_name])

                pipe_to = AutoPipelineForText2Image.from_pipe(pipe_inpaint)
                self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING[model_name])

                pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_inpaint)
                self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING[model_name])

                pipe_to = AutoPipelineForInpainting.from_pipe(pipe_inpaint)
                self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING[model_name])

                del pipe_inpaint, pipe_to
                gc.collect()

    def test_from_pipe_consistent(self):
        for model_name, model_repo in PRETRAINED_MODEL_REPO_MAPPING.items():
            if model_name in ["kandinsky", "kandinsky22"]:
                auto_pipes = [AutoPipelineForText2Image, AutoPipelineForImage2Image]
            else:
                auto_pipes = [AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting]

            # test from_pretrained
            for pipe_from_class in auto_pipes:
                pipe_from = pipe_from_class.from_pretrained(model_repo, variant="fp16", torch_dtype=torch.float16)
                pipe_from_config = dict(pipe_from.config)

                for pipe_to_class in auto_pipes:
                    pipe_to = pipe_to_class.from_pipe(pipe_from)
                    self.assertEqual(dict(pipe_to.config), pipe_from_config)

                del pipe_from, pipe_to
                gc.collect()

    def test_controlnet(self):
        # test from_pretrained
        model_repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        controlnet_repo = "lllyasviel/sd-controlnet-canny"

        controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=torch.float16)

        pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
            model_repo, controlnet=controlnet, torch_dtype=torch.float16
        )
        self.assertIsInstance(pipe_txt2img, AUTO_TEXT2IMAGE_PIPELINES_MAPPING["stable-diffusion-controlnet"])

        pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
            model_repo, controlnet=controlnet, torch_dtype=torch.float16
        )
        self.assertIsInstance(pipe_img2img, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["stable-diffusion-controlnet"])

        pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
            model_repo, controlnet=controlnet, torch_dtype=torch.float16
        )
        self.assertIsInstance(pipe_inpaint, AUTO_INPAINT_PIPELINES_MAPPING["stable-diffusion-controlnet"])

        # test from_pipe
        for pipe_from in [pipe_txt2img, pipe_img2img, pipe_inpaint]:
            pipe_to = AutoPipelineForText2Image.from_pipe(pipe_from)
            self.assertIsInstance(pipe_to, AUTO_TEXT2IMAGE_PIPELINES_MAPPING["stable-diffusion-controlnet"])
            self.assertEqual(dict(pipe_to.config), dict(pipe_txt2img.config))

            pipe_to = AutoPipelineForImage2Image.from_pipe(pipe_from)
            self.assertIsInstance(pipe_to, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["stable-diffusion-controlnet"])
            self.assertEqual(dict(pipe_to.config), dict(pipe_img2img.config))

            pipe_to = AutoPipelineForInpainting.from_pipe(pipe_from)
            self.assertIsInstance(pipe_to, AUTO_INPAINT_PIPELINES_MAPPING["stable-diffusion-controlnet"])
            self.assertEqual(dict(pipe_to.config), dict(pipe_inpaint.config))
