# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Backends that execute a resolved generation: an in-process pipeline, or an SGLang / vLLM-Omni server."""

from __future__ import annotations

import base64
import inspect
import io
import mimetypes
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import wave
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from huggingface_hub.utils import httpx
from PIL import Image

from ...modular_pipelines import ModularPipeline
from ...utils import export_to_video, load_image, load_video, logging, numpy_to_pil
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ..run import (
    PNG_COMPRESS_LEVEL,
    _as_audio_arrays,
    _as_pil_list,
    _get_generator,
    _load_pipeline,
    _save_audio_arrays,
)
from .generations import Generation
from .manifest import GMSError, ServedModel, blocks_signature


logger = logging.get_logger("diffusers-cli/serve")

# Request fields both engines accept on their OpenAI-compatible image and video routes, named as the
# pipeline kwargs they correspond to.
_ENGINE_KWARGS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "image",
        "width",
        "height",
        "num_frames",
        "fps",
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "generator",
        "num_images_per_prompt",
    }
)
_ENGINE_FIELDS = {"num_images_per_prompt": "n"}
_ENGINE_INSTALL = {"sglang": 'pip install "sglang[diffusion]"', "vllm": "pip install vllm-omni"}
_POLL_SECONDS = 1.0

_OUTPUT_FIELDS = {"image": ("images",), "video": ("frames", "videos"), "audio": ("audios", "audio")}


def backend_kwargs(backend: str, signature: inspect.Signature) -> set[str]:
    names = set(signature.parameters) - {"self"}
    if backend == "diffusers":
        return names
    return names & _ENGINE_KWARGS


def backend_output_types(backend: str) -> set[str]:
    if backend == "diffusers":
        return {"image", "video", "audio"}
    return {"image", "video"}


def backend_capabilities(args: Namespace, pipeline_cls: type) -> list[str]:
    capabilities = ["sse", "wait"]
    if args.backend != "diffusers":
        return capabilities
    # Swapping adapter weights retraces a compiled denoiser on every request, and LoRAs passed at
    # startup are part of the deployment rather than something a request may replace.
    if hasattr(pipeline_cls, "load_lora_weights") and args.compile is None and not args.lora:
        capabilities.append("adapters")
    return capabilities


def _media_bytes(name: str, value: dict[str, Any], pointer: str) -> bytes:
    try:
        if "base64" in value:
            return base64.b64decode(value["base64"], validate=True)
        response = httpx.get(value["url"], follow_redirects=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content
    except (ValueError, httpx.HTTPError) as e:
        raise GMSError("invalid_request", f"could not read input {name!r}: {e}", pointer) from e


def _media_suffix(value: dict[str, Any], default: str) -> str:
    if value.get("media_type"):
        return mimetypes.guess_extension(value["media_type"]) or default
    return Path(urlparse(value.get("url", "")).path).suffix or default


def _load_media(name: str, value: dict[str, Any], pointer: str) -> Any:
    content = _media_bytes(name, value, pointer)
    try:
        if value["type"] in ("image", "mask"):
            return load_image(Image.open(io.BytesIO(content)))
        if value["type"] == "audio":
            import torchaudio

            return torchaudio.load(io.BytesIO(content))
        with tempfile.NamedTemporaryFile(suffix=_media_suffix(value, ".mp4")) as file:
            file.write(content)
            file.flush()
            return load_video(file.name)
    except Exception as e:
        raise GMSError("invalid_request", f"could not decode input {name!r}: {e}", pointer) from e


def _save_images(images: Any, directory: Path, name: str, media_type: str, rate: float) -> list[dict[str, Any]]:
    if isinstance(images, np.ndarray):
        images = numpy_to_pil(images)
    images = _as_pil_list(images)
    if images is None:
        raise GMSError("internal", f"cannot encode output {name!r} as {media_type}: not a list of images")

    artifacts = []
    for index, image in enumerate(images):
        filename = f"{name}-{index}.{media_type.split('/')[1]}"
        if media_type == "image/jpeg":
            image = image.convert("RGB")
        if media_type == "image/png":
            image.save(directory / filename, compress_level=PNG_COMPRESS_LEVEL)
        else:
            image.save(directory / filename)
        artifacts.append({"file": filename, "media_type": media_type, "width": image.width, "height": image.height})
    return artifacts


def _save_videos(videos: Any, directory: Path, name: str, media_type: str, rate: float) -> list[dict[str, Any]]:
    artifacts = []
    for index, frames in enumerate(videos):
        frames = list(frames)
        filename = f"{name}-{index}.mp4"
        export_to_video(frames, str(directory / filename), fps=rate)
        first = frames[0]
        width, height = first.size if isinstance(first, Image.Image) else (first.shape[1], first.shape[0])
        artifacts.append(
            {
                "file": filename,
                "media_type": media_type,
                "width": width,
                "height": height,
                "duration": round(len(frames) / rate, 2),
                "fps": rate,
            }
        )
    return artifacts


def _save_audio(audios: Any, directory: Path, name: str, media_type: str, rate: float) -> list[dict[str, Any]]:
    if isinstance(audios, torch.Tensor):
        audios = audios.detach().to(torch.float32).cpu().numpy()
    arrays = _as_audio_arrays(audios)
    if arrays is None:
        raise GMSError("internal", f"cannot encode output {name!r} as {media_type}: not an audio array")

    staging = directory / f"{name}-staging"
    paths = _save_audio_arrays(arrays, int(rate), Namespace(output=str(staging) + os.sep))
    artifacts = []
    for index, path in enumerate(paths):
        filename = f"{name}-{index}.wav"
        Path(path).rename(directory / filename)
        with wave.open(str(directory / filename), "rb") as file:
            duration = round(file.getnframes() / file.getframerate(), 2)
        artifacts.append({"file": filename, "media_type": media_type, "duration": duration, "sample_rate": int(rate)})
    staging.rmdir()
    return artifacts


_SAVERS = {"image": _save_images, "video": _save_videos, "audio": _save_audio}


class DiffusersBackend:
    def __init__(self, args: Namespace, served: ServedModel):
        self._args = args
        self._served = served
        self.pipeline = _load_pipeline(args)
        self._modular = isinstance(self.pipeline, ModularPipeline)
        if self._modular:
            parameters = blocks_signature(self.pipeline.blocks).parameters
        else:
            parameters = inspect.signature(type(self.pipeline).__call__).parameters
        self._defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        self._reports_steps = "callback_on_step_end" in parameters

    def close(self) -> None:
        pass

    def failure(self) -> str | None:
        return None

    def generate(self, generation: Generation) -> dict[str, list[dict[str, Any]]]:
        request = generation.request
        kwargs = dict(request.kwargs)
        for kwarg, (name, value) in request.media.items():
            if isinstance(value, list):
                loaded = [_load_media(name, item, f"/inputs/{name}/{index}") for index, item in enumerate(value)]
            else:
                loaded = [_load_media(name, value, f"/inputs/{name}")]
            if self._served.tasks[request.task]["inputs"][name]["type"] == "audio":
                if kwarg == "initial_audio_waveforms":
                    kwargs.setdefault("initial_audio_sampling_rate", loaded[0][1])
                loaded = [waveform for waveform, _ in loaded]
            kwargs[kwarg] = loaded if isinstance(value, list) else loaded[0]

        if generation.seed is not None:
            device = self.pipeline.device.type if hasattr(self.pipeline, "device") else "cpu"
            kwargs["generator"] = _get_generator(generation.seed, device)

        if self._reports_steps:

            def on_step_end(pipeline, step, timestep, callback_kwargs):
                total_steps = getattr(pipeline, "num_timesteps", None)
                if total_steps:
                    generation.report_progress(step + 1, total_steps)
                if generation.cancel_requested:
                    pipeline._interrupt = True
                return callback_kwargs

            kwargs["callback_on_step_end"] = on_step_end

        try:
            self._load_adapters(request.adapters)
            result = self.pipeline(**kwargs)
        finally:
            if request.adapters:
                self.pipeline.unload_lora_weights()
        if generation.cancel_requested:
            return {}

        outputs = {}
        for name, spec in self._served.tasks[request.task]["outputs"].items():
            kind = spec["type"]
            # A modular pipeline returns its `PipelineState`; a standard one returns an output object.
            if self._modular:
                fields = (result.get(field) for field in _OUTPUT_FIELDS.get(kind, ()))
            else:
                fields = (getattr(result, field, None) for field in _OUTPUT_FIELDS.get(kind, ()))
            value = next((field for field in fields if field is not None), None)
            if value is None:
                continue
            rate = self._fps(kwargs) if kind == "video" else (self._args.sampling_rate or 16000)
            outputs[name] = _SAVERS[kind](value, generation.directory, name, request.output_formats[kind], rate)
        if not outputs:
            declared = list(self._served.tasks[request.task]["outputs"])
            raise GMSError(
                "internal",
                f"{type(self.pipeline).__name__} returned none of the outputs {declared} declared for task "
                f"{request.task!r}",
            )
        return outputs

    def _fps(self, kwargs: dict[str, Any]) -> float:
        for kwarg in ("fps", "frame_rate"):
            rate = kwargs.get(kwarg) or self._defaults.get(kwarg)
            if rate:
                return rate
        return self._args.fps

    def _load_adapters(self, adapters: list[dict[str, Any]]) -> None:
        if not adapters:
            return
        names = []
        weights = []
        for index, adapter in enumerate(adapters):
            name = f"request_{index}"
            try:
                self.pipeline.load_lora_weights(adapter["path"].removeprefix("hf:"), adapter_name=name)
            except Exception as e:
                raise GMSError(
                    "invalid_request",
                    f"could not load adapter {adapter['path']!r}: {type(e).__name__}: {e}",
                    f"/adapters/{index}/path",
                ) from e
            scale = adapter.get("scale", 1.0)
            targets = adapter.get("targets")
            if targets:
                valid_targets = self._served.adapters[adapter["type"]].get("targets", [])
                weights.append({target: targets.get(target, scale) for target in valid_targets})
            else:
                weights.append(scale)
            names.append(name)
        self.pipeline.set_adapters(names, adapter_weights=weights)


class EngineBackend:
    def __init__(self, engine: str, served: ServedModel, client: Any, process: subprocess.Popen | None = None):
        self._engine = engine
        self._served = served
        self._client = client
        self._process = process

    @classmethod
    def launch(cls, args: Namespace, served: ServedModel) -> EngineBackend:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        if args.backend == "sglang":
            command = ["sglang", "serve", "--model-path", args.model]
        else:
            command = ["vllm", "serve", args.model, "--omni"]
        command += ["--host", "127.0.0.1", "--port", str(port)]
        if args.revision:
            command += ["--revision", args.revision]
        command += shlex.split(args.backend_args or "")

        if shutil.which(command[0]) is None:
            raise SystemExit(
                f"--backend {args.backend} needs the `{command[0]}` executable on PATH. "
                f"Install it with: {_ENGINE_INSTALL[args.backend]}"
            )

        logger.info(f"starting {args.backend} backend: {shlex.join(command)}")
        process = subprocess.Popen(command)
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=None)
        backend = cls(args.backend, served, client, process)
        try:
            backend._wait_until_ready(command)
        except BaseException:
            backend.close()
            raise
        return backend

    def failure(self) -> str | None:
        if self._process is None or self._process.poll() is None:
            return None
        return f"the {self._engine} backend exited with code {self._process.returncode}"

    def close(self) -> None:
        self._client.close()
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def generate(self, generation: Generation) -> dict[str, list[dict[str, Any]]]:
        request = generation.request
        outputs = {spec["type"]: name for name, spec in self._served.tasks[request.task]["outputs"].items()}

        fields = {_ENGINE_FIELDS.get(kwarg, kwarg): value for kwarg, value in request.kwargs.items()}
        if generation.seed is not None:
            fields["seed"] = generation.seed
        width = fields.pop("width", None)
        height = fields.pop("height", None)
        if (width is None) != (height is None):
            missing = "width" if width is None else "height"
            raise GMSError(
                "invalid_request",
                f"the {self._engine} backend takes width and height together; {missing} is missing",
                f"/parameters/{missing}",
            )
        if width is not None:
            fields["size"] = f"{width}x{height}"

        images = []
        if "image" in request.media:
            name, value = request.media["image"]
            for index, item in enumerate(value if isinstance(value, list) else [value]):
                pointer = f"/inputs/{name}/{index}" if isinstance(value, list) else f"/inputs/{name}"
                content = _media_bytes(name, item, pointer)
                images.append((f"image-{index}{_media_suffix(item, '.png')}", content, item.get("media_type")))

        if "video" in outputs:
            return {outputs["video"]: self._generate_video(generation, fields, images)}
        return {outputs["image"]: self._generate_images(generation, fields, images)}

    def _generate_images(self, generation: Generation, fields: dict[str, Any], images: list) -> list[dict[str, Any]]:
        media_type = generation.request.output_formats["image"]
        extension = media_type.split("/")[1]
        fields = {**fields, "response_format": "b64_json", "output_format": extension}
        if images:
            response = self._client.post("/v1/images/edits", files=self._form(fields, "image", images))
        else:
            response = self._client.post("/v1/images/generations", json=fields)

        artifacts = []
        for index, item in enumerate(self._json(response)["data"]):
            content = base64.b64decode(item["b64_json"])
            filename = f"image-{index}.{extension}"
            (generation.directory / filename).write_bytes(content)
            width, height = Image.open(io.BytesIO(content)).size
            artifacts.append({"file": filename, "media_type": media_type, "width": width, "height": height})
        return artifacts

    def _generate_video(self, generation: Generation, fields: dict[str, Any], images: list) -> list[dict[str, Any]]:
        job = self._json(self._client.post("/v1/videos", files=self._form(fields, "input_reference", images[:1])))
        while job["status"] in ("queued", "in_progress"):
            if generation.cancel_requested:
                self._client.delete(f"/v1/videos/{job['id']}")
                return []
            time.sleep(_POLL_SECONDS)
            job = self._json(self._client.get(f"/v1/videos/{job['id']}"))
        if job["status"] != "completed":
            message = (job.get("error") or {}).get("message", f"video job ended with status {job['status']!r}")
            raise GMSError("internal", f"{self._engine} backend: {message}")

        response = self._client.get(f"/v1/videos/{job['id']}/content")
        if response.status_code >= 400:
            self._json(response)
        filename = "video-0.mp4"
        (generation.directory / filename).write_bytes(response.content)
        self._client.delete(f"/v1/videos/{job['id']}")

        artifact = {"file": filename, "media_type": "video/mp4"}
        if job.get("fps"):
            artifact["fps"] = job["fps"]
        if job.get("duration_s"):
            artifact["duration"] = job["duration_s"]
        return [artifact]

    @staticmethod
    def _form(fields: dict[str, Any], file_field: str, files: list) -> list:
        # Both engines read these routes as multipart forms, so plain fields are sent as nameless
        # parts: httpx falls back to urlencoding when a request carries no file parts.
        parts = []
        for key, value in fields.items():
            text = str(value).lower() if isinstance(value, bool) else str(value)
            parts.append((key, (None, text)))
        for filename, content, media_type in files:
            parts.append((file_field, (filename, content, media_type or "application/octet-stream")))
        return parts

    def _json(self, response: Any) -> Any:
        if response.status_code >= 400:
            code = "invalid_request" if response.status_code < 500 else "internal"
            raise GMSError(code, f"{self._engine} backend returned {response.status_code}: {response.text[:500]}")
        return response.json()

    def _wait_until_ready(self, command: list[str]) -> None:
        while True:
            if self._process.poll() is not None:
                raise SystemExit(
                    f"`{shlex.join(command)}` exited with code {self._process.returncode} before becoming ready."
                )
            try:
                if self._client.get("/health", timeout=5.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
