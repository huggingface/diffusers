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

"""The backend that executes a resolved generation in the server process."""

from __future__ import annotations

import base64
import contextlib
import inspect
import io
import ipaddress
import mimetypes
import socket
import tempfile
import wave
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import torch
from huggingface_hub.utils import httpx
from PIL import Image

from ...modular_pipelines import InputParam, LoopSequentialPipelineBlocks, ModularPipeline, ModularPipelineBlocks
from ...utils import export_to_video, load_image, load_video, logging, numpy_to_pil
from ...utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from ..run import _as_audio_arrays, _as_pil_list, _get_generator, _load_pipeline
from .manifest import OUTPUT_FIELDS, GMSError, Manifest, blocks_signature
from .scheduler import GenerationJob


logger = logging.get_logger("diffusers-cli/serve")

# zlib level for PNG outputs. Pillow's default of 6 takes close to four times as long to encode a
# 1024x1024 image for a file about 7% smaller; the output is lossless either way.
PNG_COMPRESS_LEVEL = 1
# Caps each media URL download, like vLLM's `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB`, so a request can't make the server read
# an unbounded file into memory.
MAX_MEDIA_DOWNLOAD_BYTES = 256 * 2**20
_SAMPLE_RATE_ATTRIBUTES = (
    "sample_rate",
    "vocoder.config.output_sampling_rate",
    "vocoder.config.sampling_rate",
    "audio_sampling_rate",
    "sound_sampling_rate",
    "vae.config.sampling_rate",
)


def backend_capabilities(args: Namespace, pipeline_cls: type) -> list[str]:
    capabilities = ["sse", "wait"]
    # Swapping adapter weights retraces a compiled denoiser on every request, and LoRAs passed at
    # startup are part of the deployment rather than something a request may replace.
    if hasattr(pipeline_cls, "load_lora_weights") and args.compile is None and not args.lora:
        capabilities.append("adapters")
    return capabilities


def _load_media(name: str, value: dict[str, Any], pointer: str, allowed_domains: list[str] | None) -> Any:
    try:
        if "base64" in value:
            content = base64.b64decode(value["base64"], validate=True)
        else:
            url = urlparse(value["url"])
            if allowed_domains is not None:
                if url.hostname not in allowed_domains:
                    raise ValueError(f"host {url.hostname!r} is not in --allowed-media-domains {allowed_domains}")
            else:
                # Only public addresses are fetched, so a request can't reach the server's own network, e.g. a cloud
                # metadata endpoint. The operator vouches for the hosts on an allowlist.
                default_port = 443 if url.scheme == "https" else 80
                for info in socket.getaddrinfo(url.hostname, url.port or default_port):
                    if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
                        raise ValueError(
                            f"host {url.hostname!r} resolves to the non-public address {info[4][0]}. Pass "
                            "--allowed-media-domains to fetch from it."
                        )
            # Redirects are not followed, so a public URL can't send the server to an address on its own network.
            with httpx.stream(
                "GET", value["url"], follow_redirects=False, timeout=DIFFUSERS_REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content += chunk
                    if len(content) > MAX_MEDIA_DOWNLOAD_BYTES:
                        raise ValueError(f"the download is larger than {MAX_MEDIA_DOWNLOAD_BYTES // 2**20} MB")
    except (ValueError, OSError, httpx.HTTPError) as e:
        raise GMSError("invalid_request", f"could not read input {name!r}: {e}", pointer) from e

    try:
        if value["type"] in ("image", "mask"):
            return load_image(Image.open(io.BytesIO(content)))
        if value["type"] == "audio":
            import torchaudio

            return torchaudio.load(io.BytesIO(content))
        if value.get("media_type"):
            suffix = mimetypes.guess_extension(value["media_type"]) or ".mp4"
        else:
            suffix = Path(urlparse(value.get("url", "")).path).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix) as file:
            file.write(content)
            file.flush()
            return load_video(file.name)
    except Exception as e:
        raise GMSError("invalid_request", f"could not decode input {name!r}: {e}", pointer) from e


def _sample_rate(pipeline: Any) -> int:
    for path in _SAMPLE_RATE_ATTRIBUTES:
        value = pipeline
        for attribute in path.split("."):
            value = getattr(value, attribute, None)
        if value:
            return int(value)
    raise SystemExit(
        f"Cannot serve audio outputs: {type(pipeline).__name__} exposes no output sample rate. Looked for "
        f"{list(_SAMPLE_RATE_ATTRIBUTES)} on the pipeline."
    )


class _Canceled(BaseException):
    """Stops a modular pipeline from its step callback. Not an `Exception`, so the blocks don't log it as an error."""


# TODO(DN6): remove once modular pipelines stream their steps with `pipe.stream()` (#14159).
class _StepCallback(ModularPipelineBlocks):
    """Calls `callback_on_step_end` after each denoising step of a modular pipeline, as a `DiffusionPipeline` does.

    Modular pipelines have no `num_timesteps` attribute, so the step count is passed in `callback_kwargs`.
    """

    @property
    def inputs(self):
        return [InputParam("callback_on_step_end", type_hint=Callable), InputParam("timesteps")]

    def __call__(self, components, block_state, i, t):
        if block_state.callback_on_step_end is not None:
            block_state.callback_on_step_end(components, i, t, {"num_timesteps": len(block_state.timesteps)})
        return components, block_state


def _denoise_loops(block: ModularPipelineBlocks) -> list[LoopSequentialPipelineBlocks]:
    loops = []
    if isinstance(block, LoopSequentialPipelineBlocks) and "denoiser" in block.sub_blocks:
        loops.append(block)
    for sub_block in getattr(block, "sub_blocks", {}).values():
        loops.extend(_denoise_loops(sub_block))
    return loops


class DiffusersBackend:
    def __init__(self, args: Namespace, manifest: Manifest):
        self.pipeline = _load_pipeline(args)
        self._allowed_media_domains = args.allowed_media_domains
        self._modular = isinstance(self.pipeline, ModularPipeline)
        if self._modular:
            # `pipeline.blocks` returns a copy, and the pipeline is already built, so the loops are edited in place.
            for loop in _denoise_loops(self.pipeline._blocks):
                loop.sub_blocks["step_callback"] = _StepCallback()
            parameters = blocks_signature(self.pipeline.blocks).parameters
        else:
            parameters = inspect.signature(type(self.pipeline).__call__).parameters
        self._reports_steps = "callback_on_step_end" in parameters
        self._sample_rate = None
        if any(spec["type"] == "audio" for task in manifest.tasks.values() for spec in task["outputs"].values()):
            self._sample_rate = _sample_rate(self.pipeline)

    def generate(self, generation: GenerationJob) -> dict[str, list[dict[str, Any]]]:
        """Call the pipeline with the request's kwargs and media, and save its outputs."""
        request = generation.request
        kwargs = dict(request.kwargs)
        for kwarg, (name, kind, value) in request.media.items():
            if isinstance(value, list):
                loaded = [
                    _load_media(name, item, f"/inputs/{name}/{index}", self._allowed_media_domains)
                    for index, item in enumerate(value)
                ]
            else:
                loaded = [_load_media(name, value, f"/inputs/{name}", self._allowed_media_domains)]
            if kind == "audio":
                if kwarg == "initial_audio_waveforms":
                    kwargs.setdefault("initial_audio_sampling_rate", loaded[0][1])
                loaded = [waveform for waveform, _ in loaded]
            kwargs[kwarg] = loaded if isinstance(value, list) else loaded[0]

        device = self.pipeline.device if hasattr(self.pipeline, "device") else torch.device("cpu")
        if request.seed is not None:
            # A CUDA generator must sit on the pipeline's own GPU, which is `cuda:1` and up with --data-parallel.
            kwargs["generator"] = _get_generator(request.seed, str(device) if device.type == "cuda" else device.type)

        if self._reports_steps:

            def on_step_end(pipeline, step, timestep, callback_kwargs):
                total_steps = callback_kwargs.get("num_timesteps", getattr(pipeline, "num_timesteps", None))
                if total_steps:
                    generation.report_progress(step + 1, total_steps)
                if generation.cancel_requested:
                    # A block can't end the loop it runs in, so a modular pipeline is stopped by raising.
                    if self._modular:
                        raise _Canceled
                    pipeline._interrupt = True
                return callback_kwargs

            kwargs["callback_on_step_end"] = on_step_end

        names = [f"request_{index}" for index in range(len(request.adapters))]
        try:
            for index, adapter in enumerate(request.adapters):
                try:
                    self.pipeline.load_lora_weights(adapter["path"].removeprefix("hf:"), adapter_name=names[index])
                except Exception as e:
                    raise GMSError(
                        "invalid_request",
                        f"could not load adapter {adapter['path']!r}: {type(e).__name__}: {e}",
                        f"/adapters/{index}/path",
                    ) from e
            if names:
                self.pipeline.set_adapters(names, adapter_weights=[adapter["weight"] for adapter in request.adapters])
            # Worker threads start on GPU 0, and with --data-parallel this pipeline may sit on another one.
            with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
                result = self.pipeline(**kwargs)
        except _Canceled:
            return {}
        finally:
            if request.adapters:
                self.pipeline.unload_lora_weights()
        if generation.cancel_requested:
            return {}

        outputs = {}
        for name, (kind, media_type) in request.outputs.items():
            value = None
            for field in OUTPUT_FIELDS[kind]:
                # A modular pipeline returns its `PipelineState`; a standard one returns an output object.
                value = result.get(field) if self._modular else getattr(result, field, None)
                if value is not None:
                    break
            if value is None:
                continue

            artifacts = []
            if kind == "image":
                images = _as_pil_list(numpy_to_pil(value) if isinstance(value, np.ndarray) else value)
                if images is None:
                    raise GMSError("internal", f"cannot encode output {name!r} as {media_type}: not a list of images")
                for index, image in enumerate(images):
                    filename = f"{name}-{index}.{media_type.split('/')[1]}"
                    if media_type == "image/jpeg":
                        image = image.convert("RGB")
                    if media_type == "image/png":
                        image.save(generation.directory / filename, compress_level=PNG_COMPRESS_LEVEL)
                    else:
                        image.save(generation.directory / filename)
                    artifacts.append(
                        {"file": filename, "media_type": media_type, "width": image.width, "height": image.height}
                    )
            elif kind == "video":
                for index, frames in enumerate(value):
                    frames = list(frames)
                    filename = f"{name}-{index}.mp4"
                    export_to_video(frames, str(generation.directory / filename), fps=request.fps)
                    first = frames[0]
                    width, height = first.size if isinstance(first, Image.Image) else (first.shape[1], first.shape[0])
                    artifacts.append(
                        {
                            "file": filename,
                            "media_type": media_type,
                            "width": width,
                            "height": height,
                            "duration": round(len(frames) / request.fps, 2),
                            "fps": request.fps,
                        }
                    )
            else:
                if isinstance(value, torch.Tensor):
                    value = value.detach().to(torch.float32).cpu().numpy()
                arrays = _as_audio_arrays(value)
                if arrays is None:
                    raise GMSError("internal", f"cannot encode output {name!r} as {media_type}: not an audio array")
                # TODO(DN6): write the files with `export_to_wav` once it is added to `diffusers.utils`.
                for index, audio in enumerate(arrays):
                    data = np.asarray(audio)
                    if data.dtype.kind == "f":
                        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
                    else:
                        data = data.astype(np.int16)
                    # `wave` takes interleaved samples x channels; the shorter axis is taken to be the channels.
                    if data.ndim == 1:
                        data = data[:, None]
                    elif data.shape[0] < data.shape[1]:
                        data = data.T
                    filename = f"{name}-{index}.wav"
                    with wave.open(str(generation.directory / filename), "wb") as file:
                        file.setnchannels(data.shape[1])
                        file.setsampwidth(2)
                        file.setframerate(self._sample_rate)
                        file.writeframes(data.tobytes())
                    artifacts.append(
                        {
                            "file": filename,
                            "media_type": media_type,
                            "duration": round(data.shape[0] / self._sample_rate, 2),
                            "sample_rate": self._sample_rate,
                        }
                    )
            outputs[name] = artifacts
        if not outputs:
            raise GMSError(
                "internal",
                f"{type(self.pipeline).__name__} returned none of the outputs {list(request.outputs)} declared for "
                f"task {request.task!r}",
            )
        return outputs
