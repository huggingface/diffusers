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

import inspect
from collections import OrderedDict

from ..configuration_utils import ConfigMixin
from .controlnet import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
)
from .deepfloyd_if import IFImg2ImgPipeline, IFInpaintingPipeline, IFPipeline
from .kandinsky import KandinskyImg2ImgPipeline, KandinskyInpaintPipeline, KandinskyPipeline
from .kandinsky2_2 import KandinskyV22Img2ImgPipeline, KandinskyV22InpaintPipeline, KandinskyV22Pipeline
from .stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from .stable_diffusion_xl import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)


AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionPipeline),
        ("stable-diffusion-xl", StableDiffusionXLPipeline),
        ("if", IFPipeline),
        ("kandinsky", KandinskyPipeline),
        ("kdnsinskyv22", KandinskyV22Pipeline),
        ("controlnet", StableDiffusionControlNetPipeline),
    ]
)

AUTO_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", StableDiffusionXLImg2ImgPipeline),
        ("if", IFImg2ImgPipeline),
        ("kandinsky", KandinskyImg2ImgPipeline),
        ("kdnsinskyv22", KandinskyV22Img2ImgPipeline),
        ("controlnet", StableDiffusionControlNetImg2ImgPipeline),
    ]
)

AUTO_INPAINTING_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", StableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", StableDiffusionXLInpaintPipeline),
        ("if", IFInpaintingPipeline),
        ("kandinsky", KandinskyInpaintPipeline),
        ("kdnsinskyv22", KandinskyV22InpaintPipeline),
        ("controlnet", StableDiffusionControlNetInpaintPipeline),
    ]
)

SUPPORTED_TASKS_MAPPINGS = [
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINTING_PIPELINES_MAPPING,
]


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


def _get_signature_keys(obj):
    parameters = inspect.signature(obj.__init__).parameters
    required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
    optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
    expected_modules = set(required_parameters.keys()) - {"self"}
    return expected_modules, optional_parameters


class AutoPipelineForText2Image(ConfigMixin):
    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        if "controlnet" in kwargs:
            text_2_image_cls = AUTO_TEXT2IMAGE_PIPELINES_MAPPING["controlnet"]

        else:
            config = cls.load_config(pretrained_model_or_path)
            text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, config["_class_name"])

        return text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        text_2_image_cls = _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING, original_cls_name)

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = _get_signature_keys(text_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        text_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in text_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(text_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {text_2_image_cls} expected {expected_modules}, but only {set(passed_class_obj.keys()) + set(original_class_obj.keys())} were passed"
            )

        model = text_2_image_cls(**text_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model


class AutoPipelineForImage2Image(ConfigMixin):
    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        if "controlnet" in kwargs:
            image_2_image_cls = AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["controlnet"]

        else:
            config = cls.load_config(pretrained_model_or_path)
            image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, config["_class_name"])

        return image_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        image_2_image_cls = _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, original_cls_name)

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = _get_signature_keys(image_2_image_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config attribute that were not expected by original pipeline is stored as its private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        image_2_image_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in image_2_image_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(image_2_image_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {image_2_image_cls} expected {expected_modules}, but only {set(passed_class_obj.keys()) + set(original_class_obj.keys())} were passed"
            )

        model = image_2_image_cls(**image_2_image_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model


class AutoPipelineForInpainting(ConfigMixin):
    config_name = "model_index.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_pipe(pipeline)` methods."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        if "controlnet" in kwargs:
            inpainting_cls = AUTO_INPAINTING_PIPELINES_MAPPING["controlnet"]

        else:
            config = cls.load_config(pretrained_model_or_path)
            inpainting_cls = _get_task_class(AUTO_INPAINTING_PIPELINES_MAPPING, config["_class_name"])

        return inpainting_cls.from_pretrained(pretrained_model_or_path, **kwargs)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        original_config = dict(pipeline.config)
        original_cls_name = pipeline.__class__.__name__

        # derive the pipeline class to instantiate
        inpainting_cls = _get_task_class(AUTO_INPAINTING_PIPELINES_MAPPING, original_cls_name)

        # define expected module and optional kwargs given the pipeline signature
        expected_modules, optional_kwargs = _get_signature_keys(inpainting_cls)

        pretrained_model_name_or_path = original_config.pop("_name_or_path", None)

        # allow users pass modules in `kwargs` to override the original pipeline's components
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        original_class_obj = {
            k: pipeline.components[k]
            for k, v in pipeline.components.items()
            if k in expected_modules and k not in passed_class_obj
        }

        # allow users pass optional kwargs to override the original pipelines config attribute
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        original_pipe_kwargs = {
            k: original_config[k]
            for k, v in original_config.items()
            if k in optional_kwargs and k not in passed_pipe_kwargs
        }

        # config that were not expected by original pipeline is stored as private attribute
        # we will pass them as optional arguments if they can be accepted by the pipeline
        additional_pipe_kwargs = [
            k[1:]
            for k in original_config.keys()
            if k.startswith("_") and k[1:] in optional_kwargs and k[1:] not in passed_pipe_kwargs
        ]
        for k in additional_pipe_kwargs:
            original_pipe_kwargs[k] = original_config.pop(f"_{k}")

        inpainting_kwargs = {**passed_class_obj, **original_class_obj, **passed_pipe_kwargs, **original_pipe_kwargs}

        # store unused config as private attribute
        unused_original_config = {
            f"{'' if k.startswith('_') else '_'}{k}": original_config[k]
            for k, v in original_config.items()
            if k not in inpainting_kwargs
        }

        missing_modules = set(expected_modules) - set(pipeline._optional_components) - set(inpainting_kwargs.keys())

        if len(missing_modules) > 0:
            raise ValueError(
                f"Pipeline {inpainting_cls} expected {expected_modules}, but only {set(passed_class_obj.keys()) + set(original_class_obj.keys())} were passed"
            )

        model = inpainting_cls(**inpainting_kwargs)
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.register_to_config(**unused_original_config)

        return model
