# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Image-generation subcommands: text-to-image, image-to-image, inpaint."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, _SubParsersAction

from diffusers.utils import load_image

from .. import BaseDiffusersCLICommand
from . import _common


def register(subparsers: _SubParsersAction) -> None:
    Text2ImageCommand.register_subcommand(subparsers)
    Image2ImageCommand.register_subcommand(subparsers)
    InpaintCommand.register_subcommand(subparsers)


def _build_call_kwargs(args: Namespace, pipeline) -> dict:
    kwargs: dict = {}
    if args.prompt is not None:
        kwargs["prompt"] = args.prompt
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.num_inference_steps is not None:
        kwargs["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        kwargs["guidance_scale"] = args.guidance_scale
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    if args.num_images != 1:
        kwargs["num_images_per_prompt"] = args.num_images

    generator = _common.get_generator(args.seed, pipeline.device.type)
    if generator is not None:
        kwargs["generator"] = generator

    kwargs.update(_common.parse_pipeline_kwargs(args.pipeline_kwargs))
    return kwargs


def _save_images(images, task: str, args: Namespace) -> list[str]:
    paths = _common.default_output_paths(task, len(images), args.output, ext="png")
    saved: list[str] = []
    for image, path in zip(images, paths):
        image.save(path)
        saved.append(str(path))
    return saved


class _BaseImageCommand(BaseDiffusersCLICommand):
    task: str = ""
    auto_cls: str = ""

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if _common.maybe_submit_remote(self.args, self.task):
            return

        pipeline = _common.load_pipeline(self.args, self.auto_cls)
        call_kwargs = _build_call_kwargs(self.args, pipeline)
        self._attach_inputs(call_kwargs)

        result = pipeline(**call_kwargs)
        saved = _save_images(result.images, self.task, self.args)
        pushed = _common.push_outputs(self.args, saved, self.task)

        _common.format_result(
            self.args,
            {
                "task": self.task,
                "model": self.args.model,
                "device": pipeline.device.type,
                "outputs": saved,
                "pushed": pushed,
                "seed": self.args.seed,
            },
        )

    def _attach_inputs(self, call_kwargs: dict) -> None:  # noqa: B027
        """Hook for subclasses to attach image/mask conditioning."""


class Text2ImageCommand(_BaseImageCommand):
    task = "text-to-image"
    auto_cls = "AutoPipelineForText2Image"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "text-to-image",
            help="Generate an image from a text prompt.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)
        parser.set_defaults(func=Text2ImageCommand)


class Image2ImageCommand(_BaseImageCommand):
    task = "image-to-image"
    auto_cls = "AutoPipelineForImage2Image"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "image-to-image",
            help="Transform an input image conditioned on a text prompt.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)

        parser.add_argument("--image", required=True, help="Path or URL to the conditioning image.")
        parser.add_argument("--strength", type=float, default=None, help="How much to transform the input (0-1).")
        parser.set_defaults(func=Image2ImageCommand)

    def _attach_inputs(self, call_kwargs: dict) -> None:
        call_kwargs["image"] = load_image(self.args.image)
        if self.args.strength is not None:
            call_kwargs["strength"] = self.args.strength


class InpaintCommand(_BaseImageCommand):
    task = "inpaint"
    auto_cls = "AutoPipelineForInpainting"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "inpaint",
            help="Inpaint a region of an image defined by a mask.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)
        parser.add_argument("--image", required=True, help="Path or URL to the base image.")
        parser.add_argument("--mask", required=True, help="Path or URL to the mask image (white=inpaint).")
        parser.add_argument("--strength", type=float, default=None, help="Strength of the inpainting transform (0-1).")
        parser.set_defaults(func=InpaintCommand)

    def _attach_inputs(self, call_kwargs: dict) -> None:
        call_kwargs["image"] = load_image(self.args.image)
        call_kwargs["mask_image"] = load_image(self.args.mask)
        if self.args.strength is not None:
            call_kwargs["strength"] = self.args.strength
