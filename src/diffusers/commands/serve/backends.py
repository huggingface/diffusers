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

"""The backends that execute a resolved generation: in the server process, or in one worker process per GPU.

`DiffusersBackend` runs the pipeline in the server process. `ParallelBackend` serves `--context-parallel`: one worker
process per GPU, all running every generation together. The server process never touches CUDA. It spawns the workers,
sends each generation to all of them over a pipe, and reads the progress and result back from rank 0. Idle workers
wait on their pipe rather than inside a collective, so the process group cannot time out between requests.
"""

from __future__ import annotations

import base64
import contextlib
import inspect
import io
import mimetypes
import multiprocessing
import os
import signal
import socket
import tempfile
import threading
import wave
from argparse import Namespace
from multiprocessing.connection import Connection, wait
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
    _as_audio_arrays,
    _as_pil_list,
    _get_generator,
    _load_pipeline,
    _write_wav,
)
from .manifest import GMSError, Manifest, ResolvedRequest, blocks_signature
from .scheduler import GenerationJob


logger = logging.get_logger("diffusers-cli/serve")

# zlib level for PNG outputs. Pillow's default of 6 takes close to four times as long to encode a
# 1024x1024 image for a file about 7% smaller; the output is lossless either way.
PNG_COMPRESS_LEVEL = 1
_OUTPUT_FIELDS = {"image": ("images",), "video": ("frames", "videos"), "audio": ("audios", "audio")}
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
    # startup are part of the deployment rather than something a request may replace. Under context
    # parallelism every rank would have to swap them in lockstep, which is untested.
    if (
        hasattr(pipeline_cls, "load_lora_weights")
        and args.compile is None
        and not args.lora
        and not args.context_parallel
    ):
        capabilities.append("adapters")
    return capabilities


def _load_media(name: str, value: dict[str, Any], pointer: str) -> Any:
    try:
        if "base64" in value:
            content = base64.b64decode(value["base64"], validate=True)
        else:
            response = httpx.get(value["url"], follow_redirects=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.content
    except (ValueError, httpx.HTTPError) as e:
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

    artifacts = []
    for index, audio in enumerate(arrays):
        filename = f"{name}-{index}.wav"
        _write_wav(audio, int(rate), directory / filename)
        with wave.open(str(directory / filename), "rb") as file:
            duration = round(file.getnframes() / file.getframerate(), 2)
        artifacts.append({"file": filename, "media_type": media_type, "duration": duration, "sample_rate": int(rate)})
    return artifacts


_SAVERS = {"image": _save_images, "video": _save_videos, "audio": _save_audio}


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


def load_request_media(request: ResolvedRequest) -> dict[str, Any]:
    """The pipeline kwargs of a request, with its media inputs fetched and decoded."""
    kwargs = dict(request.kwargs)
    for kwarg, (name, kind, value) in request.media.items():
        if isinstance(value, list):
            loaded = [_load_media(name, item, f"/inputs/{name}/{index}") for index, item in enumerate(value)]
        else:
            loaded = [_load_media(name, value, f"/inputs/{name}")]
        if kind == "audio":
            if kwarg == "initial_audio_waveforms":
                kwargs.setdefault("initial_audio_sampling_rate", loaded[0][1])
            loaded = [waveform for waveform, _ in loaded]
        kwargs[kwarg] = loaded if isinstance(value, list) else loaded[0]
    return kwargs


class DiffusersBackend:
    def __init__(self, args: Namespace, manifest: Manifest, save_outputs: bool = True):
        # Under context parallelism every rank runs the call, and only rank 0 writes the files.
        self._save_outputs = save_outputs
        self.pipeline = _load_pipeline(args)
        self._modular = isinstance(self.pipeline, ModularPipeline)
        if self._modular:
            parameters = blocks_signature(self.pipeline.blocks).parameters
        else:
            parameters = inspect.signature(type(self.pipeline).__call__).parameters
        self._reports_steps = "callback_on_step_end" in parameters
        outputs = (spec["type"] for task in manifest.tasks.values() for spec in task["outputs"].values())
        self._sample_rate = _sample_rate(self.pipeline) if "audio" in outputs else None

    def close(self) -> None:
        pass

    def failure(self) -> str | None:
        return None

    def generate(self, generation: GenerationJob) -> dict[str, list[dict[str, Any]]]:
        return self.run(load_request_media(generation.request), generation)

    def run(self, kwargs: dict[str, Any], generation: Any) -> dict[str, list[dict[str, Any]]]:
        """Call the pipeline with the request's kwargs, media already loaded, and save its outputs."""
        request = generation.request
        device = self.pipeline.device if hasattr(self.pipeline, "device") else torch.device("cpu")
        if request.seed is not None:
            # A CUDA generator must sit on the pipeline's own GPU, which is `cuda:1` and up with --data-parallel.
            kwargs["generator"] = _get_generator(request.seed, str(device) if device.type == "cuda" else device.type)

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
            # Worker threads start on GPU 0, and with --data-parallel this pipeline may sit on another one.
            with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
                result = self.pipeline(**kwargs)
        finally:
            if request.adapters:
                self.pipeline.unload_lora_weights()
        # Read first: under context parallelism `cancel_requested` is a collective every rank must join.
        if generation.cancel_requested or not self._save_outputs:
            return {}

        outputs = {}
        for name, (kind, media_type) in request.outputs.items():
            # A modular pipeline returns its `PipelineState`; a standard one returns an output object.
            if self._modular:
                fields = (result.get(field) for field in _OUTPUT_FIELDS.get(kind, ()))
            else:
                fields = (getattr(result, field, None) for field in _OUTPUT_FIELDS.get(kind, ()))
            value = next((field for field in fields if field is not None), None)
            if value is None:
                continue
            rate = request.fps if kind == "video" else self._sample_rate
            outputs[name] = _SAVERS[kind](value, generation.directory, name, media_type, rate)
        if not outputs:
            raise GMSError(
                "internal",
                f"{type(self.pipeline).__name__} returned none of the outputs {list(request.outputs)} declared for "
                f"task {request.task!r}",
            )
        return outputs

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
            weights.append(adapter["weight"])
            names.append(name)
        self.pipeline.set_adapters(names, adapter_weights=weights)


_POLL_SECONDS = 0.1
_SHUTDOWN_SECONDS = 30.0


class _WorkerJob:
    """The parts of a generation `DiffusersBackend.run` reads, inside a worker.

    `cancel_requested` is a collective: rank 0 reads the flag the server sets and broadcasts it, so every rank
    interrupts at the same step. A rank that stopped alone would leave the others waiting in the next collective.
    """

    def __init__(self, job: dict[str, Any], rank: int, conn: Connection, cancel_flag: Any):
        self.request = job["request"]
        self.directory = Path(job["directory"])
        self._rank = rank
        self._conn = conn
        self._cancel_flag = cancel_flag

    def report_progress(self, step: int, total_steps: int) -> None:
        if self._rank == 0:
            self._conn.send(("progress", step, total_steps))

    @property
    def cancel_requested(self) -> bool:
        flag = torch.tensor([self._cancel_flag.value if self._rank == 0 else 0], dtype=torch.uint8)
        torch.distributed.broadcast(flag, src=0)
        return bool(flag.item())


def _serve_jobs(backend: Any, rank: int, conn: Connection, cancel_flag: Any) -> None:
    while True:
        job = conn.recv()
        if job is None:
            return
        generation = _WorkerJob(job, rank, conn, cancel_flag)
        try:
            reply = ("complete", backend.run(job["kwargs"], generation))
        except GMSError as e:
            reply = ("failed", e.code, e.message, e.pointer)
        # The helpers shared with `run` raise SystemExit, which must not end the worker.
        except (Exception, SystemExit) as e:
            logger.error(f"generation failed on rank {rank}", exc_info=True)
            reply = ("failed", "internal", f"{type(e).__name__}: {e}", None)
        if rank == 0:
            conn.send(reply)


def _worker_main(
    rank: int, world_size: int, port: int, args: Namespace, manifest: Manifest, conn: Connection, cancel_flag: Any
) -> None:
    # The server process decides when workers stop, so Ctrl-C in the terminal must not interrupt them mid-call.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    if args.log_level is not None:
        logging.set_verbosity(logging.get_log_levels_dict()[args.log_level])
    # Loading applies `--context-parallel`, which joins the process group of all ranks.
    backend = DiffusersBackend(args, manifest, save_outputs=rank == 0)
    conn.send(("ready",))
    try:
        _serve_jobs(backend, rank, conn, cancel_flag)
    finally:
        torch.distributed.destroy_process_group()


class ParallelBackend:
    def __init__(self, args: Namespace, manifest: Manifest):
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise SystemExit(f"--context-parallel needs at least 2 visible GPUs, found {world_size}.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        self._failure: str | None = None
        self._failed = threading.Event()
        self._closing = False
        # Spawned rather than forked: a forked child inherits whatever CUDA state its parent has.
        context = multiprocessing.get_context("spawn")
        # Lock-free: a worker killed while holding the lock would leave the server blocked on its next write.
        self._cancel_flag = context.Value("b", 0, lock=False)
        self._conns: list[Connection] = []
        self._processes = []
        for rank in range(world_size):
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(rank, world_size, port, args, manifest, child_conn, self._cancel_flag),
                name=f"diffusers-serve-rank-{rank}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(process)

        logger.info(f"loading the pipeline on {world_size} GPUs...")
        self._wait_until_ready()
        threading.Thread(target=self._monitor, name="diffusers-serve-workers", daemon=True).start()

    def _wait_until_ready(self) -> None:
        waiting = dict(zip(self._conns, self._processes))
        while waiting:
            sentinels = {process.sentinel: process for process in waiting.values()}
            for ready in wait([*waiting, *sentinels]):
                process = waiting.pop(ready, None) if isinstance(ready, Connection) else sentinels[ready]
                try:
                    if isinstance(ready, Connection) and ready.recv() == ("ready",):
                        continue
                except EOFError:
                    pass
                self.close()
                raise SystemExit(
                    f"Context-parallel worker {process.name} exited with code {process.exitcode} while loading "
                    "the pipeline. Its error is in the log above."
                )

    def _monitor(self) -> None:
        sentinels = {process.sentinel: process for process in self._processes}
        dead = sentinels[wait(list(sentinels))[0]]
        if self._closing:
            return
        # `exitcode` polls without waiting, so it can still be None just after the sentinel fires.
        dead.join()
        self._failure = f"context-parallel worker {dead.name} exited with code {dead.exitcode}"
        self._failed.set()
        logger.error(f"{self._failure}; the remaining workers are stopped and the server no longer generates.")
        # The other ranks would wait forever in their next collective, so the group is not reused.
        self.close()

    def failure(self) -> str | None:
        return self._failure

    def generate(self, generation: GenerationJob) -> dict[str, list[dict[str, Any]]]:
        if self._failure is not None:
            raise GMSError("internal", self._failure)
        job = {
            "request": generation.request,
            "kwargs": load_request_media(generation.request),
            "directory": str(generation.directory),
        }
        self._cancel_flag.value = 0
        for conn in self._conns:
            conn.send(job)

        rank_0 = self._conns[0]
        while True:
            if generation.cancel_requested:
                self._cancel_flag.value = 1
            try:
                if not rank_0.poll(_POLL_SECONDS):
                    if self._failure is not None:
                        raise GMSError("internal", self._failure)
                    continue
                message = rank_0.recv()
            except (EOFError, OSError) as e:
                # Rank 0's pipe only breaks once it has exited. Only the monitor reaps workers: a second
                # thread joining the same process can read its exit code before it is recorded.
                self._failed.wait()
                raise GMSError("internal", self._failure) from e
            if message[0] == "progress":
                generation.report_progress(*message[1:])
            elif message[0] == "complete":
                return message[1]
            else:
                _, code, text, pointer = message
                raise GMSError(code, text, pointer)

    def close(self) -> None:
        self._closing = True
        for conn in self._conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for process in self._processes:
            process.join(timeout=_SHUTDOWN_SECONDS)
            if process.is_alive():
                process.kill()
                process.join()
