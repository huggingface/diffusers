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

"""Video-generation subcommands: text-to-video, image-to-video.

There is no AutoPipeline for video, so these commands load via
``DiffusionPipeline`` and rely on the repo's ``model_index.json`` to pick
the right pipeline class (CogVideoX, Hunyuan, LTX, Wan, etc.).
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, _SubParsersAction

from diffusers.utils import load_image

from .. import BaseDiffusersCLICommand
from . import _common


def register(subparsers: _SubParsersAction) -> None:
    Text2VideoCommand.register_subcommand(subparsers)
    Image2VideoCommand.register_subcommand(subparsers)


def _add_video_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to generate.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video.")


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
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames

    generator = _common.get_generator(args.seed, pipeline.device.type)
    if generator is not None:
        kwargs["generator"] = generator

    kwargs.update(_common.parse_pipeline_kwargs(args.pipeline_kwargs))
    return kwargs


def _save_video(frames, args: Namespace, task: str) -> str:
    from diffusers.utils import export_to_video

    path = _common.default_output_paths(task, 1, args.output, ext="mp4")[0]
    export_to_video(frames, str(path), fps=args.fps)
    return str(path)


class _BaseVideoCommand(BaseDiffusersCLICommand):
    task: str = ""

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if _common.maybe_submit_remote(self.args, self.task):
            return
        pipeline = _common.load_pipeline(self.args, "DiffusionPipeline")
        call_kwargs = _build_call_kwargs(self.args, pipeline)
        self._attach_inputs(call_kwargs)

        result = pipeline(**call_kwargs)
        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        out_path = _save_video(frames, self.args, self.task)
        pushed = _common.push_outputs(self.args, [out_path], self.task)

        _common.format_result(
            self.args,
            {
                "task": self.task,
                "model": self.args.model,
                "device": pipeline.device.type,
                "outputs": [out_path],
                "pushed": pushed,
                "fps": self.args.fps,
                "seed": self.args.seed,
            },
        )

    def _attach_inputs(self, call_kwargs: dict) -> None:  # noqa: B027
        """Hook for subclasses to attach conditioning inputs."""


class Text2VideoCommand(_BaseVideoCommand):
    task = "text-to-video"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "text-to-video",
            help="Generate a video clip from a text prompt.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _add_video_arguments(parser)
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)
        parser.set_defaults(func=Text2VideoCommand)


class Image2VideoCommand(_BaseVideoCommand):
    task = "image-to-video"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "image-to-video",
            help="Generate a video clip conditioned on an input image.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _add_video_arguments(parser)
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)
        parser.add_argument("--image", required=True, help="Path or URL to the conditioning image.")
        parser.set_defaults(func=Image2VideoCommand)

    def _attach_inputs(self, call_kwargs: dict) -> None:
        call_kwargs["image"] = load_image(self.args.image)
