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

"""``diffusers-cli modular`` — run a custom ModularPipeline.

Modular pipelines don't fit the ``task -> AutoPipelineFor*`` taxonomy: the
pipeline blocks themselves define the surface. This command takes free-form
``--inputs key=value`` (or a JSON blob) and forwards them to the modular
pipeline call, then auto-detects the result type so the agent doesn't need
to know whether it asked for an image, video, or audio output.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from typing import Any

from .. import BaseDiffusersCLICommand
from . import _common


def register(subparsers: _SubParsersAction) -> None:
    ModularCommand.register_subcommand(subparsers)


def _parse_inputs(args: Namespace) -> dict[str, Any]:
    """Combine ``--inputs-json`` and repeated ``--inputs key=value`` into one dict.

    Values from ``--inputs`` are JSON-decoded when possible (so booleans,
    numbers, lists, and nested objects survive); plain strings fall back to
    raw text.
    """
    out: dict[str, Any] = {}
    if args.inputs_json:
        try:
            decoded = json.loads(args.inputs_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--inputs-json must be valid JSON: {e}") from e
        if not isinstance(decoded, dict):
            raise SystemExit("--inputs-json must decode to a JSON object.")
        out.update(decoded)

    for pair in args.inputs or []:
        if "=" not in pair:
            raise SystemExit(f"--inputs entries must look like key=value, got {pair!r}.")
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _save_auto(value: Any, args: Namespace, task: str) -> list[str]:
    """Save ``value`` based on its runtime type and return the written paths."""
    pil_images = _as_pil_list(value)
    if pil_images is not None:
        paths = _common.default_output_paths(task, len(pil_images), args.output, ext="png")
        for img, path in zip(pil_images, paths):
            img.save(path)
        return [str(p) for p in paths]

    frames = _as_frame_sequence(value)
    if frames is not None:
        from diffusers.utils import export_to_video

        path = _common.default_output_paths(task, 1, args.output, ext="mp4")[0]
        export_to_video(frames, str(path), fps=args.fps)
        return [str(path)]

    audios = _as_audio_arrays(value)
    if audios is not None:
        from .audio import _save_audio

        return _save_audio(audios, args.sampling_rate or 16000, args, task)

    # Fallback: dump as JSON.
    path = _common.default_output_paths(task, 1, args.output, ext="json")[0]
    Path(path).write_text(json.dumps(value, default=str, indent=2))
    return [str(path)]


def _as_pil_list(value: Any):
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return None
    if isinstance(value, PILImage):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, PILImage) for v in value):
        return list(value)
    return None


def _as_frame_sequence(value: Any):
    """A frame sequence is a list of PIL images or numpy frames meant to be a single clip."""
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        PILImage = None  # type: ignore[assignment]

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first = value[0]
        if PILImage is not None and isinstance(first, PILImage):
            # Heuristic: distinguish "list of images we want as PNGs" from "frame sequence".
            # The modular pipeline call already returned a single value, so we treat a
            # homogeneous list of >=2 images as a clip.
            return list(value)
        try:
            import numpy as np

            if isinstance(first, np.ndarray):
                return list(value)
        except ImportError:
            pass
    return None


def _as_audio_arrays(value: Any):
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        return [value]
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, np.ndarray) for v in value)
    ):
        return list(value)
    return None


class ModularCommand(BaseDiffusersCLICommand):
    task = "modular"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "modular",
            help="Run a custom ModularPipeline with free-form inputs.",
        )
        _common.add_loading_arguments(parser)
        _common.add_optimization_arguments(parser)
        parser.add_argument(
            "--inputs",
            action="append",
            default=None,
            help='Inputs as key=value (value JSON-decoded when possible). Repeat to add multiple.',
        )
        parser.add_argument(
            "--inputs-json",
            default=None,
            help="Inputs as a single JSON object (merged with any --inputs entries).",
        )
        parser.add_argument(
            "--output-key",
            default=None,
            help='Optional intermediate to extract (e.g. "image", "video"). '
            "Forwarded to ModularPipeline as the ``output`` argument.",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=8,
            help="FPS used when the output happens to be a frame sequence.",
        )
        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=None,
            help="Sample rate used when the output happens to be an audio array.",
        )
        _common.add_remote_arguments(parser)
        _common.add_output_arguments(parser)
        parser.set_defaults(func=ModularCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if _common.maybe_submit_remote(self.args, self.task):
            return

        pipeline = self._load_modular()
        call_kwargs = _parse_inputs(self.args)
        if self.args.output_key is not None:
            call_kwargs["output"] = self.args.output_key

        result = pipeline(**call_kwargs)
        saved = _save_auto(result, self.args, self.task)
        pushed = _common.push_outputs(self.args, saved, self.task)

        _common.format_result(
            self.args,
            {
                "task": self.task,
                "model": self.args.model,
                "pipeline_class": type(pipeline).__name__,
                "outputs": saved,
                "pushed": pushed,
                "output_key": self.args.output_key,
            },
        )

    def _load_modular(self):
        from diffusers import ModularPipeline

        dtype = _common.resolve_dtype(self.args.dtype)
        device = _common.resolve_device(self.args.device)

        from_pretrained_kwargs: dict[str, Any] = {
            "trust_remote_code": self.args.trust_remote_code,
        }
        if dtype != "auto":
            from_pretrained_kwargs["torch_dtype"] = dtype
        if self.args.revision:
            from_pretrained_kwargs["revision"] = self.args.revision
        if self.args.token:
            from_pretrained_kwargs["token"] = self.args.token

        pipeline = ModularPipeline.from_pretrained(self.args.model, **from_pretrained_kwargs)
        if not hasattr(pipeline, "to"):
            return pipeline

        pipeline = _common.map_to_device(pipeline, self.args, device)
        if self.args.vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if self.args.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if self.args.attention_backend != "default":
            _common._set_attention_backend(pipeline, self.args.attention_backend)
        if self.args.context_parallel:
            _common._enable_context_parallel(pipeline)
        return pipeline
