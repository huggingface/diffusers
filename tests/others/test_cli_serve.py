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
"""Unit tests for `diffusers-cli serve`.

No weights are loaded: the backend runs a fake pipeline.
"""

import asyncio
import base64
import inspect
import io
import json
import os
import re
import shlex
import threading
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers.commands.run import _DEFAULT_REMOTE_IMAGE
from diffusers.commands.serve import ServeCommand, _shut_down_when_idle
from diffusers.commands.serve.backends import DiffusersBackend, backend_capabilities
from diffusers.commands.serve.manifest import GMSError, Manifest, blocks_signature, derive_manifest
from diffusers.commands.serve.scheduler import GenerationJob, Scheduler
from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam


MODEL_ID = "test-org/fake-model"
WAIT = {"Prefer": "wait=30"}


class FakePipeline:
    _lora_loadable_modules = ["transformer", "text_encoder"]
    device = torch.device("cpu")
    num_timesteps = 0

    def __init__(self):
        self.calls = []
        self.lora_calls = []
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.error = None
        self.result = None

    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        image: Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 4,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
        )
        self.started.set()
        self.release.wait(timeout=30)
        if self.error is not None:
            raise self.error
        self.num_timesteps = num_inference_steps
        self._interrupt = False
        for step in range(num_inference_steps):
            if self._interrupt:
                break
            callback_on_step_end(self, step, 0, {})
        if self.result is not None:
            return self.result
        size = (width or 32, height or 32)
        return SimpleNamespace(images=[Image.new("RGB", size, "red") for _ in range(num_images_per_prompt)])

    def load_lora_weights(self, repo, adapter_name=None):
        self.lora_calls.append(("load", repo, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.lora_calls.append(("set", names, adapter_weights))

    def unload_lora_weights(self):
        self.lora_calls.append(("unload",))


FAKE_SIGNATURE = inspect.signature(FakePipeline.__call__)


class FakeImageBlock(ModularPipelineBlocks):
    """A one-block modular pipeline that declares its inputs the way the built-in blocks do, mostly untyped."""

    model_name = "fake"

    @property
    def inputs(self):
        return [
            InputParam("prompt"),
            InputParam("image"),
            InputParam("width"),
            InputParam("num_inference_steps", default=4),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        size = (block_state.width or 32, 32)
        block_state.images = [Image.new("RGB", size, "red")]
        self.set_block_state(state, block_state)
        return components, state


def _args(**overrides):
    values = {
        "compile": None,
        "lora": None,
        "context_parallel": False,
    }
    return Namespace(**{**values, **overrides})


def _manifest(manifest=None, pipeline_cls=FakePipeline, signature=None, args=None):
    """A `Manifest` bound to `pipeline_cls`, from the given manifest dict or bytes, or derived from the pipeline."""
    signature = signature or inspect.signature(pipeline_cls.__call__)
    capabilities = backend_capabilities(args or _args(), pipeline_cls)
    if manifest is None:
        manifest = derive_manifest(MODEL_ID, pipeline_cls, signature, capabilities)
    raw = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    return Manifest(raw, json.loads(raw), pipeline_cls, signature, capabilities, revision="abc123")


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Build a test client around a `DiffusersBackend` running `FakePipeline`, or around the given pipeline or
    backend. Returns `(client, pipeline)`."""
    pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
    from fastapi.testclient import TestClient

    from diffusers.commands.serve.app import create_app

    def _build(manifest=None, app_options=None, pipeline=None, signature=None, backend=None, **arg_overrides):
        pipeline = pipeline or FakePipeline()
        monkeypatch.setattr("diffusers.commands.serve.backends._load_pipeline", lambda args: pipeline)
        args = _args(**arg_overrides)
        bound = _manifest(manifest=manifest, pipeline_cls=type(pipeline), signature=signature, args=args)
        backend = backend or DiffusersBackend(args, bound)
        app = create_app(bound, Scheduler([backend], str(tmp_path)), **(app_options or {}))
        return TestClient(app), pipeline

    return _build


LATTICE_MANIFEST = {
    "id": MODEL_ID,
    "modalities": {"in": ["image"], "out": ["video"]},
    "tasks": {
        "text_to_video": {
            "inputs": {
                "prompt": {"type": "text", "required": True},
                "negative_prompt": {"type": "text", "active_when": {"parameter": "guidance_scale", "gt": 1}},
            },
            "outputs": {"video": {"type": "video"}},
            "parameters": {
                "duration": {"type": "number", "min": 1, "max": 20, "lattice": {"frames": "8k+1", "fps": 24}},
                "width": {"type": "integer", "default": 768, "min": 512, "max": 1536, "multiple_of": 64},
                "steps": {"type": "integer", "default": 30, "min": 1, "max": 100},
                "fps": {"type": "number"},
                "guidance_scale": {"type": "number", "default": 3.0, "min": 1.0, "max": 10.0},
                "seed": {"type": "integer"},
            },
            "presets": {
                "default": {},
                "distilled": {"parameters": {"steps": 8, "guidance_scale": 1.0}},
            },
        }
    },
    "capabilities": ["sse", "wait"],
}


class FakeVideoPipeline:
    def __call__(
        self,
        prompt: str = None,
        negative_prompt: str | None = None,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        generator: torch.Generator | None = None,
    ):
        pass


class FakeAudioVideoPipeline(FakePipeline):
    vocoder = SimpleNamespace(config=SimpleNamespace(sampling_rate=24000))

    def __call__(self, prompt: str = None, frame_rate: float = 24.0):
        return self.result


class TestManifest:
    def test_derive_manifest_from_signature(self):
        manifest = derive_manifest(MODEL_ID, FakePipeline, FAKE_SIGNATURE, ["sse", "wait", "adapters"])

        # An optional `image` kwarg yields one task without it and one that requires it.
        assert list(manifest["tasks"]) == ["text_to_image", "image_to_image"], manifest["tasks"].keys()
        text_to_image = manifest["tasks"]["text_to_image"]
        assert text_to_image["inputs"] == {
            "prompt": {"type": "text", "required": True},
            "negative_prompt": {"type": "text"},
        }
        assert manifest["tasks"]["image_to_image"]["inputs"]["image"] == {"type": "image", "required": True}
        assert text_to_image["outputs"] == {"image": {"type": "image"}}
        # Core vocabulary names replace the pipeline's own; tensors, callbacks and output plumbing are dropped.
        assert text_to_image["parameters"] == {
            "height": {"type": "integer"},
            "width": {"type": "integer"},
            "steps": {"type": "integer", "default": 4},
            "sigmas": {"type": "array", "items": {"type": "number"}},
            "guidance_scale": {"type": "number", "default": 3.5},
            "num_outputs": {"type": "integer", "default": 1},
            "seed": {"type": "integer"},
        }
        assert text_to_image["x-diffusers"] == {
            "parameters": {"steps": "num_inference_steps", "num_outputs": "num_images_per_prompt", "seed": "generator"}
        }
        assert manifest["capabilities"] == ["sse", "wait", "adapters"]
        assert manifest["adapters"] == {"lora": {"targets": ["transformer", "text_encoder"]}}

    @pytest.mark.parametrize(
        "body, code, pointer",
        [
            ({"parameters": {"cfg": 3}}, "invalid_request", "/parameters/cfg"),
            ({"parameters": {"steps": 500}}, "invalid_request", "/parameters/steps"),
            ({"parameters": {"steps": "8"}}, "invalid_request", "/parameters/steps"),
            ({"inputs": {}}, "invalid_request", "/inputs/prompt"),
            ({"task": "image_to_video"}, "invalid_request", "/task"),
            # The lattice pins duration to 24 fps, so another fps would deliver a different length.
            ({"parameters": {"duration": 4, "fps": 12}}, "invalid_request", "/parameters/fps"),
            # `negative_prompt` is only honored while guidance_scale > 1.
            (
                {"inputs": {"prompt": "a cat", "negative_prompt": "blurry"}, "preset": "distilled"},
                "invalid_request",
                "/inputs/negative_prompt",
            ),
        ],
    )
    def test_resolve_rejects_invalid_requests(self, body, code, pointer):
        bound = _manifest(manifest=LATTICE_MANIFEST, pipeline_cls=FakeVideoPipeline)
        with pytest.raises(GMSError) as error:
            bound.resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}, **body})
        assert (error.value.code, error.value.pointer) == (code, pointer), error.value.message

    def test_resolve_rejects_non_http_media_urls(self):
        bound = _manifest()
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat", "image": {"type": "image", "url": "/etc/passwd"}}}
        with pytest.raises(GMSError) as error:
            bound.resolve(body)
        assert error.value.pointer == "/inputs/image/url", error.value.message

    @pytest.mark.parametrize(
        "path",
        ["hf:/etc/passwd", "hf:../loras/x", "hf:../outputs", "hf:detail", "/etc/passwd", "https://x/y.safetensors"],
    )
    def test_resolve_rejects_adapter_paths_that_are_not_repo_ids(self, path):
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "adapters": [{"type": "lora", "path": path}]}
        with pytest.raises(GMSError) as error:
            _manifest().resolve(body)
        assert (error.value.code, error.value.pointer) == ("invalid_request", "/adapters/0/path"), error.value.message

    def test_resolve_rejects_adapter_repo_ids_that_name_a_local_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "someone" / "detail").mkdir(parents=True)
        body = {
            "model": MODEL_ID,
            "inputs": {"prompt": "a cat"},
            "adapters": [{"type": "lora", "path": "hf:someone/detail"}],
        }
        with pytest.raises(GMSError, match="adapter path"):
            _manifest().resolve(body)

    def test_resolve_maps_request_to_pipeline_kwargs(self):
        bound = _manifest(manifest=LATTICE_MANIFEST, pipeline_cls=FakeVideoPipeline)
        resolved = bound.resolve(
            {
                "model": MODEL_ID,
                "preset": "distilled",
                "inputs": {"prompt": "a cat"},
                "parameters": {"duration": 8, "seed": 7},
            }
        )
        # 8 s at 24 fps is 192 frames; the nearest point on the 8k+1 lattice is 193. The preset overrides the
        # task defaults.
        assert resolved.kwargs == {
            "prompt": "a cat",
            "num_frames": 193,
            "frame_rate": 24,
            "width": 768,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
        }, resolved.kwargs
        assert resolved.seed == 7
        assert resolved.task == "text_to_video"

    @pytest.mark.parametrize(
        "change, args, message",
        [
            # flux2-klein's published manifest declares `guidance_type`, which its pipeline does not take.
            (
                {"parameters": {"guidance_type": {"type": "string", "enum": ["distilled"]}}},
                {},
                "declares parameter 'guidance_type' on task 'text_to_video', which FakeVideoPipeline does not take",
            ),
            ({"capabilities": ["adapters"]}, {}, "request adapters are off"),
            ({"capabilities": ["stream"]}, {}, "declares the 'stream' capability, which this server does not provide"),
            ({"parameters": {"duration": {"type": "number"}}}, {}, "without a frame lattice"),
        ],
    )
    def test_manifest_the_server_cannot_honor_fails_startup(self, change, args, message):
        manifest = json.loads(json.dumps(LATTICE_MANIFEST))
        if "parameters" in change:
            manifest["tasks"]["text_to_video"]["parameters"].update(change["parameters"])
        else:
            manifest.update(change)
        with pytest.raises(SystemExit, match=re.escape(message)):
            _manifest(manifest=manifest, pipeline_cls=FakeVideoPipeline, args=_args(**args))


class TestServeApp:
    def test_generation_lifecycle(self, serve):
        client, pipeline = serve()
        body = {
            "model": MODEL_ID,
            "inputs": {"prompt": "a cat"},
            "parameters": {"width": 64, "height": 48, "steps": 3, "seed": 11},
        }
        response = client.post("/v1/generations", json=body, headers=WAIT)
        assert response.status_code == 201, response.text
        generation = response.json()
        assert generation["status"] == "complete", generation
        assert generation["seed"] == 11
        assert generation["revision"] == "abc123"
        assert generation["progress"] is None
        assert generation["usage"]["compute_seconds"] >= 0

        call = pipeline.calls[0]
        assert (call["prompt"], call["width"], call["height"], call["num_inference_steps"]) == ("a cat", 64, 48, 3)
        assert torch.equal(call["generator"].get_state(), torch.Generator().manual_seed(11).get_state())

        (artifact,) = generation["outputs"]["image"]
        assert {k: artifact[k] for k in ("media_type", "width", "height")} == {
            "media_type": "image/png",
            "width": 64,
            "height": 48,
        }
        image = Image.open(io.BytesIO(client.get(artifact["url"]).content))
        assert (image.format, image.size) == ("PNG", (64, 48))

        events = client.get(f"/v1/generations/{generation['id']}/events").text
        assert events.count("event: progress") == 3, events
        assert 'data: {"step": 3, "total_steps": 3, "fraction": 1.0}' in events
        assert events.rstrip().split("\n\n")[-1].startswith("id: 4\nevent: complete\n")
        # Resuming after the last progress event replays only what followed it.
        resumed = client.get(f"/v1/generations/{generation['id']}/events", headers={"Last-Event-ID": "3"}).text
        assert "event: progress" not in resumed and "event: complete" in resumed, resumed

        assert client.delete(f"/v1/generations/{generation['id']}").status_code == 204
        missing = client.get(f"/v1/generations/{generation['id']}")
        assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")
        assert client.get(artifact["url"]).status_code == 404

    def test_inline_response_and_media_input(self, serve):
        client, pipeline = serve()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        body = {
            "model": MODEL_ID,
            "inputs": {
                "prompt": "a cat",
                "image": {
                    "type": "image",
                    "base64": base64.b64encode(buffer.getvalue()).decode(),
                    "media_type": "image/png",
                },
            },
            "output_format": "image/jpeg",
            "response_format": "b64_json",
        }
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation
        # The task is inferred from the inputs: only `image_to_image` accepts an image.
        assert pipeline.calls[0]["image"].size == (8, 8)
        (artifact,) = generation["outputs"]["image"]
        assert artifact["media_type"] == "image/jpeg" and "url" not in artifact
        assert Image.open(io.BytesIO(base64.b64decode(artifact["base64"]))).format == "JPEG"

    def test_joint_video_and_audio_outputs(self, serve, monkeypatch):
        monkeypatch.setattr(
            "diffusers.commands.serve.backends.export_to_video",
            lambda frames, path, fps: Path(path).write_bytes(b"mp4"),
        )
        manifest = {
            "id": MODEL_ID,
            "tasks": {
                "text_to_video": {
                    "inputs": {"prompt": {"type": "text", "required": True}},
                    "outputs": {"video": {"type": "video"}, "audio": {"type": "audio"}},
                }
            },
            "capabilities": ["wait"],
        }
        client, pipeline = serve(manifest=manifest, pipeline=FakeAudioVideoPipeline())
        pipeline.result = SimpleNamespace(
            frames=[[Image.new("RGB", (32, 24)) for _ in range(16)]],
            audio=np.zeros((1, 2, 8000), dtype=np.float32),
        )
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation

        # Video is encoded at the pipeline's default `frame_rate`, audio at its vocoder's sample rate.
        (video,) = generation["outputs"]["video"]
        (audio,) = generation["outputs"]["audio"]
        assert {k: v for k, v in video.items() if k != "url"} == {
            "media_type": "video/mp4",
            "width": 32,
            "height": 24,
            "duration": 0.67,
            "fps": 24.0,
        }
        assert {k: v for k, v in audio.items() if k != "url"} == {
            "media_type": "audio/wav",
            "duration": 0.33,
            "sample_rate": 24000,
        }
        assert client.get(audio["url"]).content[:4] == b"RIFF"

    def test_failed_generation_returns_error_envelope(self, serve):
        client, pipeline = serve()
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        envelope = {"error": {"code": "internal", "message": "RuntimeError: CUDA out of memory"}}

        pipeline.error = RuntimeError("CUDA out of memory")
        created = client.post("/v1/generations", json=body)
        assert created.status_code == 201
        # The event stream closes once the generation is done, so reading it to the end waits for the failure.
        events = client.get(f"/v1/generations/{created.json()['id']}/events").text
        assert 'event: error\ndata: {"code": "internal", "message": "RuntimeError: CUDA out of memory"}' in events
        response = client.get(f"/v1/generations/{created.json()['id']}")
        assert (response.status_code, response.json()) == (500, envelope)
        # There is no `failed` status: a create that waited for the failure returns the envelope as well.
        waited = client.post("/v1/generations", json=body, headers=WAIT)
        assert (waited.status_code, waited.json()) == (500, envelope)

        # A SystemExit from the backend must not end the worker: the next generation still runs.
        pipeline.error = SystemExit("unsupported")
        assert client.post("/v1/generations", json=body, headers=WAIT).status_code == 500
        pipeline.error = None
        assert client.post("/v1/generations", json=body, headers=WAIT).json()["status"] == "complete"

    def test_undeclared_capabilities_are_off(self, serve):
        manifest = json.loads(json.dumps(LATTICE_MANIFEST))
        manifest["tasks"]["text_to_video"] = {
            "inputs": {"prompt": {"type": "text", "required": True}},
            "outputs": {"image": {"type": "image"}},
        }
        manifest["capabilities"] = []
        client, pipeline = serve(manifest=manifest)
        # The pipeline is held, so a response that waited for the generation could never return.
        pipeline.release.clear()
        created = client.post("/v1/generations", json={"model": MODEL_ID, "inputs": {"prompt": "a cat"}}, headers=WAIT)
        assert created.json()["status"] in ("queued", "generating"), "without `wait`, Prefer: wait is ignored"
        events = client.get(f"/v1/generations/{created.json()['id']}/events")
        assert (events.status_code, events.json()["error"]["code"]) == (501, "not_implemented")
        pipeline.release.set()

    def test_failed_backend_is_unhealthy(self, serve):
        backend = SimpleNamespace(failure=lambda: "context-parallel worker rank-1 exited with code 3")
        client, _ = serve(backend=backend)
        response = client.get("/health")
        expected = {"status": "unhealthy", "reason": "context-parallel worker rank-1 exited with code 3"}
        assert (response.status_code, response.json()) == (503, expected)

    def test_health_checks_do_not_count_as_activity(self, serve):
        client, _ = serve()
        client.app.state.last_request = 0.0
        client.get("/health")
        assert client.app.state.last_request == 0.0, "a health check must not keep an idle server alive"
        client.get("/v1/models")
        assert client.app.state.last_request > 0.0, "a request should reset the idle clock"

    def test_public_url_replaces_the_request_address_in_output_links(self, serve):
        client, _ = serve(app_options={"public_url": "https://proxy.example/sandbox/proxy/8000/"})
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        expected = f"https://proxy.example/sandbox/proxy/8000/v1/generations/{generation['id']}/files/image-0.png"
        assert generation["outputs"]["image"][0]["url"] == expected

        # A reverse proxy may forward requests with the published path still attached.
        for path in ("/v1/models", "/sandbox/proxy/8000/v1/models"):
            assert client.get(path).status_code == 200, path
        prefixed = client.post("/sandbox/proxy/8000/v1/generations", json=body, headers=WAIT)
        assert prefixed.status_code == 201, prefixed.text

    def test_cancel_queued_and_running(self, serve):
        client, pipeline = serve()
        pipeline.release.clear()
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        running = client.post("/v1/generations", json=body).json()
        queued = client.post("/v1/generations", json=body).json()
        assert queued["status"] == "queued", queued
        assert pipeline.started.wait(timeout=30), "the worker never picked up the first generation"

        assert client.post(f"/v1/generations/{queued['id']}/cancel").json()["status"] == "canceled"
        client.post(f"/v1/generations/{running['id']}/cancel")
        pipeline.release.set()

        # The stream ends once the generation is done; a canceled generation emits no `complete` event.
        events = client.get(f"/v1/generations/{running['id']}/events").text
        assert "event: complete" not in events, events
        canceled = client.get(f"/v1/generations/{running['id']}").json()
        assert (canceled["status"], canceled["outputs"]) == ("canceled", None)
        assert len(pipeline.calls) == 1, "the canceled queued generation must never reach the pipeline"

    def test_request_adapters_are_loaded_then_unloaded(self, serve):
        client, pipeline = serve()
        body = {
            "model": MODEL_ID,
            "inputs": {"prompt": "a cat"},
            "adapters": [
                {"type": "lora", "path": "hf:someone/style", "scale": 0.8},
                {"type": "lora", "path": "hf:someone/detail", "scale": 0.5, "targets": {"text_encoder": 0.1}},
            ],
        }
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation
        assert pipeline.lora_calls == [
            ("load", "someone/style", "request_0"),
            ("load", "someone/detail", "request_1"),
            ("set", ["request_0", "request_1"], [0.8, {"transformer": 0.5, "text_encoder": 0.1}]),
            ("unload",),
        ]

    def test_discovery_serves_manifest_bytes_verbatim(self, serve):
        manifest = derive_manifest(MODEL_ID, FakePipeline, FAKE_SIGNATURE, ["sse", "wait", "adapters"])
        raw = json.dumps(manifest, indent=3).encode()
        client, _ = serve(manifest=raw)

        response = client.get(f"/v1/models/{MODEL_ID}")
        assert raw in response.content
        discovery = response.json()
        assert discovery["spec"] == "gms/0.1"
        assert discovery["manifest"] == manifest
        assert "server" not in discovery, "the manifest describes the server, so nothing is added to it"
        assert client.get("/v1/models").json() == {
            "data": [
                {
                    "id": MODEL_ID,
                    "modalities": {"in": ["image"], "out": ["image"]},
                    "tasks": ["text_to_image", "image_to_image"],
                }
            ]
        }


class TestModularPipeline:
    def test_generation_runs_a_modular_pipeline(self, serve):
        pipeline = FakeImageBlock().init_pipeline()
        signature = blocks_signature(pipeline.blocks)
        manifest = derive_manifest(MODEL_ID, type(pipeline), signature, ["wait"])
        assert list(manifest["tasks"]) == ["text_to_image", "image_to_image"], manifest["tasks"].keys()
        assert manifest["tasks"]["text_to_image"]["inputs"] == {"prompt": {"type": "text", "required": True}}
        assert manifest["tasks"]["text_to_image"]["parameters"] == {
            "steps": {"type": "integer", "default": 4},
            "seed": {"type": "integer"},
        }

        client, _ = serve(manifest=manifest, pipeline=pipeline, signature=signature)

        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "parameters": {"seed": 3}}
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation
        assert generation["seed"] == 3
        (artifact,) = generation["outputs"]["image"]
        assert (artifact["media_type"], artifact["width"], artifact["height"]) == ("image/png", 32, 32)


class FakeJobsApi:
    def __init__(self, logs=("server started",)):
        self.logs = logs
        self.run_kwargs = None
        self.cancelled = False

    def run_job(self, **kwargs):
        self.run_kwargs = kwargs
        return SimpleNamespace(id="job-1", owner=SimpleNamespace(name="me"))

    def fetch_job_logs(self, **kwargs):
        for line in self.logs:
            if isinstance(line, BaseException):
                raise line
            yield line

    def inspect_job(self, **kwargs):
        return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED", expose_urls=None))

    def cancel_job(self, **kwargs):
        self.cancelled = True


class TestServeRemote:
    def _run(self, monkeypatch, api, argv):
        monkeypatch.setattr("diffusers.commands.serve.remote.HfApi", lambda token: api)
        monkeypatch.setattr("diffusers.commands.serve.remote.get_token", lambda: "hf_x")
        parser = ArgumentParser()
        ServeCommand.register_subcommand(parser.add_subparsers())
        ServeCommand(parser.parse_args(["serve", "org/model", "--remote", *argv])).run()

    def test_remote_starts_the_server_in_a_job(self, monkeypatch):
        api = FakeJobsApi()
        argv = ["--dtype", "fp16", "--flavor", "a10g-large", "--port", "9000"]
        self._run(monkeypatch, api, [*argv, "--dependencies", "extra-package"])

        job = api.run_kwargs
        assert (job["flavor"], job["image"]) == ("a10g-large", _DEFAULT_REMOTE_IMAGE), job
        assert job["expose"] == [8000] and job["timeout"] == "24h", job
        assert job["secrets"] == {"HF_TOKEN": "hf_x"}, "gated models need the HF token inside the Job"

        assert job["command"][:2] == ["sh", "-c"], job["command"]
        install, start = job["command"][2].split(" && ")
        assert "fastapi" in install and "extra-package" in install, install
        start = shlex.split(start)
        # The server runs under the interpreter the dependencies were installed into, not whatever is on PATH.
        assert start[:4] == ["exec", "$(uv python find --system)", "-m", "diffusers.commands.diffusers_cli"], start
        serve_argv = start[start.index("serve") + 1 :]
        options = dict(zip(serve_argv[::2], serve_argv[1::2]))
        assert options["--model"] == "org/model" and options["--dtype"] == "fp16", options
        # The server binds the exposed port on every interface, and builds output links on the public address.
        assert (options["--host"], options["--port"]) == ("0.0.0.0", "8000"), options
        assert options["--public-url"] == "https://${JOB_ID}--8000.hf.jobs", options
        assert options["--idle-timeout"] == "30m", "a remote server should shut itself down when idle by default"
        assert not {"--remote", "--flavor", "--dependencies"} & set(serve_argv), serve_argv
        assert not api.cancelled, "a Job that completed on its own needs no cancel"

    def test_interrupt_cancels_the_job(self, monkeypatch):
        api = FakeJobsApi(logs=("server started", KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            self._run(monkeypatch, api, [])
        assert api.cancelled, "Ctrl-C must cancel the Job so it stops billing"


class BlockingBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def failure(self):
        return None

    def generate(self, generation):
        self.started.set()
        self.release.wait(timeout=30)
        return {}


class TestScheduler:
    def test_each_backend_runs_its_own_generation(self, tmp_path):
        backends = [BlockingBackend(), BlockingBackend()]
        scheduler = Scheduler(backends, str(tmp_path))
        request = _manifest().resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}})
        jobs = [GenerationJob(request) for _ in backends]
        for job in jobs:
            scheduler.submit(job)

        for index, backend in enumerate(backends):
            assert backend.started.wait(timeout=30), f"backend {index} never picked up a generation"
        assert [job.status for job in jobs] == ["generating", "generating"], "both generations should run at once"

        for backend in backends:
            backend.release.set()
        for job in jobs:
            asyncio.run(job.wait(timeout=30))
        assert [job.status for job in jobs] == ["complete", "complete"]
        assert not scheduler.busy(), "the scheduler should be idle once every backend has finished"

    def test_generation_past_the_timeout_fails(self, tmp_path):
        class SteppingBackend:
            """Stops at the next step once interrupted, like a pipeline's step callback."""

            def failure(self):
                return None

            def generate(self, generation):
                while not generation.cancel_requested:
                    time.sleep(0.01)
                return {}

        scheduler = Scheduler([SteppingBackend()], str(tmp_path), timeout=0.1)
        job = GenerationJob(_manifest().resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}}))
        scheduler.submit(job)
        asyncio.run(job.wait(timeout=30))
        assert job.status == "failed", job.status
        assert "--generation-timeout" in job.error.message, job.error.message

    def test_generation_stuck_past_the_timeout_fails_health(self, tmp_path):
        backend = BlockingBackend()
        scheduler = Scheduler([backend], str(tmp_path), timeout=60)
        job = GenerationJob(_manifest().resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}}))
        scheduler.submit(job)
        assert backend.started.wait(timeout=30), "the worker never picked up the generation"
        assert scheduler.failure() is None, "a generation within its time limit is healthy"

        # As if it had run for over twice the limit without reaching a step that could stop it.
        job.deadline -= 121
        assert "did not stop when interrupted" in (scheduler.failure() or ""), scheduler.failure()
        backend.release.set()


class FakeClock:
    """Stands in for the `time` module: `sleep` advances the clock instead of waiting."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if self.now > 1000:
            raise AssertionError("the idle watcher never shut the server down")
        self.now += seconds


class TestIdleShutdown:
    def _watch(self, monkeypatch, busy_checks):
        """Run the watcher with a 1m timeout, checked every 30s, until it shuts the server down; returns when it did."""
        clock = FakeClock()
        monkeypatch.setattr("diffusers.commands.serve.time", clock)
        busy = iter(busy_checks)
        server = SimpleNamespace(should_exit=False)
        app = SimpleNamespace(state=SimpleNamespace(last_request=0.0))
        _shut_down_when_idle(server, app, SimpleNamespace(busy=lambda: next(busy, False)), "1m")
        assert server.should_exit, "the watcher should only return once it told the server to exit"
        return clock.now

    def test_idle_server_shuts_down(self, monkeypatch):
        assert self._watch(monkeypatch, []) == 60, "an idle server should exit after one timeout"

    def test_busy_server_stays_up_until_the_work_ends(self, monkeypatch):
        # Busy at the checks at 30s and 60s, so the idle clock restarts at 60s.
        assert self._watch(monkeypatch, [True, True]) == 120, "the server should exit one timeout after the work ends"


class TestServeCommand:
    def _parse(self, argv):
        parser = ArgumentParser()
        ServeCommand.register_subcommand(parser.add_subparsers())
        return parser.parse_args(["serve", *argv])

    def test_model_is_positional_and_loads_in_bf16(self, monkeypatch):
        pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
        resolved = {}

        def stop(args):
            resolved.update(vars(args))
            raise RuntimeError("stop after the flag check")

        monkeypatch.setattr("diffusers.commands.serve._resolve_pipeline", stop)
        with pytest.raises(RuntimeError, match="stop after the flag check"):
            ServeCommand(self._parse(["org/model"])).run()
        assert resolved["model"] == "org/model", resolved
        assert resolved["dtype"] == "bf16", "serve should default to bf16 rather than `run`'s fp32 `auto`"


def _fake_worker_main(rank, world_size, port, args, manifest, conn, cancel_flag):
    """Stands in for `_worker_main` in forked children: no pipeline, no process group."""
    if args.fail_at_startup:
        os._exit(3)
    conn.send(("ready",))
    while True:
        job = conn.recv()
        if job is None:
            return
        if job["request"].seed == -1:
            os._exit(4)
        if rank != 0:
            continue
        conn.send(("progress", 1, 2))
        if job["kwargs"]["prompt"] == "cancel me":
            # The server raises the flag once it has read the progress above.
            for _ in range(1000):
                if cancel_flag.value:
                    break
                time.sleep(0.01)
            conn.send(("complete", {} if cancel_flag.value else {"image": []}))
        elif job["kwargs"]["prompt"] == "bad":
            conn.send(("failed", "invalid_request", "bad prompt", "/inputs/prompt"))
        else:
            conn.send(("complete", {"image": [{"file": "image-0.png", "world_size": world_size}]}))


class FakeGeneration:
    def __init__(self, prompt="a cat", seed=0):
        self.request = SimpleNamespace(kwargs={"prompt": prompt}, media={}, task="text_to_image", seed=seed)
        self.directory = Path("/tmp/unused")
        self.cancel_requested = False
        self.progress = []

    def report_progress(self, step, total_steps):
        self.progress.append((step, total_steps))
        if self.request.kwargs["prompt"] == "cancel me":
            self.cancel_requested = True


class TestParallelBackend:
    @pytest.fixture
    def backend(self, monkeypatch):
        import multiprocessing

        from diffusers.commands.serve import backends

        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("the fake workers are forked so they can use a test-local target")
        fork = multiprocessing.get_context("fork")
        monkeypatch.setattr(backends.multiprocessing, "get_context", lambda method: fork)
        monkeypatch.setattr(backends, "_worker_main", _fake_worker_main)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        built = []

        def _build(**arg_overrides):
            args = Namespace(**{"fail_at_startup": False, **arg_overrides})
            built.append(backends.ParallelBackend(args, _manifest()))
            return built[-1]

        yield _build
        for backend in built:
            backend.close()

    def test_generation_runs_on_every_rank_and_rank_0_answers(self, backend):
        generation = FakeGeneration()
        outputs = backend().generate(generation)
        assert outputs == {"image": [{"file": "image-0.png", "world_size": 2}]}, outputs
        assert generation.progress == [(1, 2)], "rank 0's progress should reach the generation"

    def test_worker_error_becomes_a_request_error(self, backend):
        with pytest.raises(GMSError, match="bad prompt") as error:
            backend().generate(FakeGeneration(prompt="bad"))
        assert (error.value.code, error.value.pointer) == ("invalid_request", "/inputs/prompt")

    def test_cancel_reaches_the_workers(self, backend):
        outputs = backend().generate(FakeGeneration(prompt="cancel me"))
        assert outputs == {}, "a cancel raised mid-generation should reach the workers"

    def test_dead_worker_fails_the_server(self, backend):
        server = backend()
        with pytest.raises(GMSError, match="exited"):
            server.generate(FakeGeneration(seed=-1))
        assert "exited with code 4" in server.failure(), server.failure()
        with pytest.raises(GMSError, match="exited with code 4"):
            server.generate(FakeGeneration())

    def test_worker_dying_at_startup_stops_the_server(self, backend):
        with pytest.raises(SystemExit, match="exited with code 3 while loading the pipeline"):
            backend(fail_at_startup=True)

    def test_worker_arguments_survive_spawn(self):
        import pickle

        parser = ArgumentParser()
        ServeCommand.register_subcommand(parser.add_subparsers())
        args = parser.parse_args(["serve", "org/model", "--context-parallel"])
        # Spawned workers receive the parsed flags and the manifest pickled.
        restored_args, restored_manifest = pickle.loads(pickle.dumps((args, _manifest())))
        assert restored_args.context_parallel and restored_manifest.tasks


class TestParallelWorker:
    def _jobs(self, backend, rank, *jobs):
        from multiprocessing import Pipe, Value

        from diffusers.commands.serve.backends import _serve_jobs

        server, worker = Pipe()
        for job in (*jobs, None):
            server.send(job)
        _serve_jobs(backend, rank, worker, Value("b", 0))
        replies = []
        while server.poll():
            replies.append(server.recv())
        return replies

    def _job(self, prompt):
        return {"request": None, "kwargs": {"prompt": prompt}, "directory": "/tmp/unused"}

    def test_only_rank_0_reports_progress_and_results(self):
        def run(kwargs, generation):
            generation.report_progress(1, 1)
            if kwargs["prompt"] == "bad":
                raise GMSError("invalid_request", "bad prompt", "/inputs/prompt")
            if kwargs["prompt"] == "crash":
                raise SystemExit("unsupported")
            return {"image": []}

        backend = SimpleNamespace(run=run)
        replies = self._jobs(backend, 0, self._job("ok"), self._job("bad"), self._job("crash"))
        assert replies == [
            ("progress", 1, 1),
            ("complete", {"image": []}),
            ("progress", 1, 1),
            ("failed", "invalid_request", "bad prompt", "/inputs/prompt"),
            ("progress", 1, 1),
            ("failed", "internal", "SystemExit: unsupported", None),
        ], "a failed generation must not end the worker"
        assert self._jobs(backend, 1, self._job("ok"), self._job("bad")) == [], "only rank 0 talks to the server"

    def test_cancel_is_broadcast_from_rank_0(self, monkeypatch):
        from multiprocessing import Value

        from diffusers.commands.serve.backends import _WorkerJob

        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(_free_port()))
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
        try:
            flag = Value("b", 0)
            generation = _WorkerJob(self._job("ok"), 0, None, flag)
            assert not generation.cancel_requested
            flag.value = 1
            assert generation.cancel_requested, "rank 0 should broadcast the flag the server raised"
        finally:
            torch.distributed.destroy_process_group()


def _free_port():
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
