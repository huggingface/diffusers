# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from ..configuration_utils import ConfigMixin
from collections import OrderedDict

from .stable_diffusion import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline,
)
from .stable_diffusion_xl import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from .deepfloyd_if import IFPipeline, IFImg2ImgPipeline, IFInpaintingPipeline
from .kandinsky import KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyInpaintPipeline
from .kandinsky2_2 import KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline, KandinskyV22InpaintPipeline

from .pipeline_utils import DiffusionPipeline


AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionPipeline),
        ("stable-diffusion-xl", StableDiffusionXLPipeline),
        ("if", IFPipeline),
        ("kandinsky", KandinskyPipeline),
        ("kdnsinskyv22", KandinskyV22Pipeline),
    ]
)

AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", StableDiffusionXLImg2ImgPipeline),
        ("if", IFImg2ImgPipeline),
        ("kandinsky", KandinskyImg2ImgPipeline),
        ("kdnsinskyv22", KandinskyV22Img2ImgPipeline),
    ]
)

AUTO_INPAINTING_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", StableDiffusionXLInpaintPipeline),
        ("if", IFInpaintingPipeline),
        ("kandinsky", KandinskyInpaintPipeline),
        ("kdnsinskyv22", KandinskyV22InpaintPipeline),
    ]
)

SUPPORTED_TASKS_MAPPINGS = [AUTO_TEXT2IMAGE_PIPELINES_MAPPING, AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, AUTO_INPAINTING_PIPELINES_MAPPING]


def _get_task_class(mapping, pipeline_class_name):

    def get_model(pipeline_class_name):

        for task_mapping in SUPPORTED_TASKS_MAPPINGS:
            for model_name, pipeline in task_mapping.items():
                if pipeline.__name__ == pipeline_class_name:
                    return model_name

    model_name = get_model(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class
    raise ValueError(
        f"AutoPipeline can't find a pipeline linked to {pipeline_class_name} for {model_name} in {mapping}"
    )


class AutoPipelineForText2Image(ConfigMixin):

    config_name = "model_index.json"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        config = cls.load_config(pretrained_model_or_path)

        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, config["_class_name"])

        return text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)
        
    @classmethod
    def from_pipe(cls, pipeline: DiffusionPipeline, **kwargs):
        print(pipeline.config.keys())
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, pipeline.__class__.__name__)

        return text_2_image_cls(**pipeline.components)


class AutoPipelineForImage2Image(ConfigMixin):

    config_name = "model_index.json"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        config = cls.load_config(pretrained_model_or_path)

        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, config["_class_name"])

        return image_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)
        
    @classmethod
    def from_pipe(cls, pipeline: DiffusionPipeline, **kwargs):
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, pipeline.__class__.__name__)

        return image_2_image_cls(**pipeline.components, **kwargs)

class AutoPipelineForInpainting(ConfigMixin):

    config_name = "model_index.json"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        config = cls.load_config(pretrained_model_or_path)

        inpainting_cls = _get_task_class(AUTO_INPAINTING_PIPELINES_MAPPING, config["_class_name"])

        return inpainting_cls.from_pretrained(pretrained_model_or_path, **kwargs)
        
    @classmethod
    def from_pipe(cls, pipeline: DiffusionPipeline, **kwargs):
        inpainting_cls = _get_task_class(AUTO_INPAINTING_PIPELINES_MAPPING, pipeline.__class__.__name__)

        return inpainting_cls(**pipeline.components)