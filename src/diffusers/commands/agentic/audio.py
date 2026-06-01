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

"""Audio-generation subcommands: text-to-audio."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, _SubParsersAction

from .. import BaseDiffusersCLICommand
from . import _common


def register(subparsers: _SubParsersAction) -> None:
    Text2AudioCommand.register_subcommand(subparsers)


def _save_audio(audios, sampling_rate: int, args: Namespace, task: str) -> list[str]:
    """Save one or more audio arrays as WAV files."""
    import numpy as np
    from scipy.io.wavfile import write as wavfile_write

    paths = _common.default_output_paths(task, len(audios), args.output, ext="wav")
    saved: list[str] = []
    for audio, path in zip(audios, paths):
        data = np.asarray(audio)
        if data.dtype.kind == "f":
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        if data.ndim > 1 and data.shape[0] < data.shape[-1]:
            # ``(channels, samples)`` → ``(samples, channels)`` for scipy.
            data = data.T
        wavfile_write(str(path), sampling_rate, data)
        saved.append(str(path))
    return saved


class Text2AudioCommand(BaseDiffusersCLICommand):
    task = "text-to-audio"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "text-to-audio",
            help="Generate an audio clip (music or sound) from a text prompt.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        _common.add_generation_arguments(parser)
        _common.add_remote_arguments(parser)
        parser.add_argument(
            "--audio-length-in-s",
            type=float,
            default=None,
            help="Duration of the generated audio in seconds.",
        )
        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=None,
            help="Override the sampling rate written to the WAV file.",
        )
        _common.add_output_arguments(parser)
        parser.set_defaults(func=Text2AudioCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if _common.maybe_submit_remote(self.args, self.task):
            return
        pipeline = _common.load_pipeline(self.args, "DiffusionPipeline")

        call_kwargs: dict = {}
        if self.args.prompt is not None:
            call_kwargs["prompt"] = self.args.prompt
        if self.args.negative_prompt is not None:
            call_kwargs["negative_prompt"] = self.args.negative_prompt
        if self.args.num_inference_steps is not None:
            call_kwargs["num_inference_steps"] = self.args.num_inference_steps
        if self.args.guidance_scale is not None:
            call_kwargs["guidance_scale"] = self.args.guidance_scale
        if self.args.audio_length_in_s is not None:
            call_kwargs["audio_length_in_s"] = self.args.audio_length_in_s
        if self.args.num_images != 1:
            call_kwargs["num_waveforms_per_prompt"] = self.args.num_images

        generator = _common.get_generator(self.args.seed, pipeline.device.type)
        if generator is not None:
            call_kwargs["generator"] = generator

        call_kwargs.update(_common.parse_pipeline_kwargs(self.args.pipeline_kwargs))

        result = pipeline(**call_kwargs)
        audios = getattr(result, "audios", None)
        if audios is None:
            audios = result[0]

        sampling_rate = self.args.sampling_rate
        if sampling_rate is None:
            pipeline_sr = getattr(pipeline, "sampling_rate", None)
            if isinstance(pipeline_sr, int):
                sampling_rate = pipeline_sr
            else:
                vocoder_config = getattr(getattr(pipeline, "vocoder", None), "config", None)
                sampling_rate = getattr(vocoder_config, "sampling_rate", 16000) if vocoder_config else 16000

        saved = _save_audio(audios, sampling_rate, self.args, self.task)
        pushed = _common.push_outputs(self.args, saved, self.task)

        _common.format_result(
            self.args,
            {
                "task": self.task,
                "model": self.args.model,
                "device": pipeline.device.type,
                "outputs": saved,
                "pushed": pushed,
                "sampling_rate": sampling_rate,
                "seed": self.args.seed,
            },
        )
