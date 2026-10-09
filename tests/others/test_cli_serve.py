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

No weights are loaded: the diffusers backend runs a fake pipeline and the engine backend talks to a fake
SGLang / vLLM-Omni server through an httpx mock transport.
"""

import base64
import inspect
import io
import json
import subprocess
import sys
import threading
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from huggingface_hub.utils import httpx
from PIL import Image

from diffusers.commands.generate import GenerateCommand
from diffusers.commands.serve import ServeCommand
from diffusers.commands.serve.backends import (
    DiffusersBackend,
    EngineBackend,
    backend_capabilities,
    backend_kwargs,
    backend_output_types,
)
from diffusers.commands.serve.generations import Generation, GenerationQueue
from diffusers.commands.serve.manifest import GMSError, Manifest, ServedModel, blocks_signature, derive_manifest
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
    values = {"backend": "diffusers", "compile": None, "lora": None, "fps": 8, "sampling_rate": None}
    return Namespace(**{**values, **overrides})


def _served(backend="diffusers", manifest=None, pipeline_cls=FakePipeline, **arg_overrides):
    args = _args(backend=backend, **arg_overrides)
    signature = inspect.signature(pipeline_cls.__call__)
    kwarg_names = backend_kwargs(backend, signature)
    if manifest is None:
        manifest = derive_manifest(MODEL_ID, pipeline_cls, signature, kwarg_names)
    raw = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    return ServedModel(
        Manifest(raw=raw, data=json.loads(raw)),
        kwarg_names,
        backend_output_types(backend),
        backend_capabilities(args, pipeline_cls),
        revision="abc123",
    )


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Build a test client around a `DiffusersBackend` running `FakePipeline`. Returns `(client, pipeline)`."""
    pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
    from fastapi.testclient import TestClient

    from diffusers.commands.serve.app import create_app

    def _build(manifest=None, app_options=None, **arg_overrides):
        pipeline = FakePipeline()
        monkeypatch.setattr("diffusers.commands.serve.backends._load_pipeline", lambda args: pipeline)
        served = _served(manifest=manifest, **arg_overrides)
        backend = DiffusersBackend(_args(**arg_overrides), served)
        app = create_app(served, GenerationQueue(backend, str(tmp_path)), **(app_options or {}))
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
                "last_image": {"type": "image"},
            },
            "outputs": {"video": {"type": "video"}},
            "parameters": {
                "duration": {"type": "number", "min": 1, "max": 20, "lattice": {"frames": "8k+1", "fps": 24}},
                "width": {"type": "integer", "default": 768, "min": 512, "max": 1536, "multiple_of": 64},
                "steps": {"type": "integer", "default": 30, "min": 1, "max": 100},
                "fps": {"type": "number"},
                "guidance_scale": {"type": "number", "default": 3.0, "min": 1.0, "max": 10.0},
                "guidance_type": {"type": "string", "enum": ["cfg", "distilled"], "default": "cfg"},
                "seed": {"type": "integer"},
                "video_decoder": {"type": "string", "enum": ["conv", "diffusion"], "default": "conv"},
            },
            "presets": {
                "default": {},
                "distilled": {"parameters": {"steps": 8, "guidance_scale": 1.0}},
                "quality": {"parameters": {"steps": 40, "video_decoder": "diffusion"}},
            },
        }
    },
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


class TestManifest:
    def test_derive_manifest_from_signature(self):
        manifest = derive_manifest(MODEL_ID, FakePipeline, FAKE_SIGNATURE, backend_kwargs("diffusers", FAKE_SIGNATURE))

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
        assert manifest["adapters"] == {"lora": {"targets": ["transformer", "text_encoder"]}}

    @pytest.mark.parametrize(
        "body, code, pointer",
        [
            ({"parameters": {"cfg": 3}}, "invalid_request", "/parameters/cfg"),
            ({"parameters": {"width": 700}}, "invalid_request", "/parameters/width"),
            ({"parameters": {"steps": 500}}, "invalid_request", "/parameters/steps"),
            ({"parameters": {"steps": "8"}}, "invalid_request", "/parameters/steps"),
            ({"inputs": {}}, "invalid_request", "/inputs/prompt"),
            ({"inputs": {"prompt": "a cat", "caption": "x"}}, "invalid_request", "/inputs/caption"),
            ({"task": "image_to_video"}, "invalid_request", "/task"),
            ({"task": []}, "invalid_request", "/task"),
            ({"parameters": []}, "invalid_request", "/parameters"),
            ({"parameters": {"seed": 10**30}}, "invalid_request", "/parameters/seed"),
            ({"parameters": {"duration": float("nan")}}, "invalid_request", "/parameters/duration"),
            # The lattice pins duration to 24 fps, so another fps would deliver a different length.
            ({"parameters": {"duration": 4, "fps": 12}}, "invalid_request", "/parameters/fps"),
            ({"steps": 8}, "invalid_request", "/steps"),
            ({"preset": "quality"}, "invalid_request", "/preset"),
            ({"model": "someone/else"}, "not_found", None),
            ({"adapters": [{"type": "lora", "path": "hf:a/b"}]}, "not_implemented", "/adapters"),
            ({"output_format": "video/webm"}, "invalid_request", "/output_format"),
            # `negative_prompt` is only honored while guidance_scale > 1.
            (
                {"inputs": {"prompt": "a cat", "negative_prompt": "blurry"}, "preset": "distilled"},
                "invalid_request",
                "/inputs/negative_prompt",
            ),
            # `last_image` has no matching pipeline kwarg, so the server overlay disables it.
            (
                {"inputs": {"prompt": "a cat", "last_image": {"type": "image", "url": "https://x/y.png"}}},
                "invalid_request",
                "/inputs/last_image",
            ),
            # A parameter the pipeline cannot set is refused rather than dropped, unless it is the default.
            ({"parameters": {"video_decoder": "diffusion"}}, "not_implemented", "/parameters/video_decoder"),
        ],
    )
    def test_resolve_rejects_invalid_requests(self, body, code, pointer):
        served = _served(manifest=LATTICE_MANIFEST, pipeline_cls=FakeVideoPipeline)
        with pytest.raises(GMSError) as error:
            served.resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}, **body})
        assert (error.value.code, error.value.pointer) == (code, pointer), error.value.message

    def test_resolve_rejects_non_http_media_urls(self):
        served = _served()
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat", "image": {"type": "image", "url": "/etc/passwd"}}}
        with pytest.raises(GMSError) as error:
            served.resolve(body)
        assert error.value.pointer == "/inputs/image/url", error.value.message

    @pytest.mark.parametrize("path", ["hf:/etc/passwd", "hf:../loras/x", "/etc/passwd", "https://x/y.safetensors"])
    def test_resolve_rejects_adapter_paths_that_are_not_repo_ids(self, path):
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "adapters": [{"type": "lora", "path": path}]}
        with pytest.raises(GMSError) as error:
            _served().resolve(body)
        assert (error.value.code, error.value.pointer) == ("invalid_request", "/adapters/0/path"), error.value.message

    def test_resolve_maps_request_to_pipeline_kwargs(self):
        served = _served(manifest=LATTICE_MANIFEST, pipeline_cls=FakeVideoPipeline)
        resolved = served.resolve(
            {
                "model": MODEL_ID,
                "preset": "distilled",
                "inputs": {"prompt": "a cat"},
                "parameters": {"duration": 8, "seed": 7, "guidance_type": "cfg"},
            }
        )
        # 8 s at 24 fps is 192 frames; the nearest point on the 8k+1 lattice is 193. The preset overrides the
        # task defaults, and `guidance_type` at its default is accepted although the pipeline has no such kwarg.
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

    def test_discovery_overlay(self):
        served = _served(manifest=LATTICE_MANIFEST, pipeline_cls=FakeVideoPipeline)
        assert served.overlay() == {
            "tasks": {"text_to_video": {"presets": ["default", "distilled"]}},
            "capabilities": ["sse", "wait"],
            "limits": {"text_to_video": {"inputs": {"last_image": False}}},
        }


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
        }
        client, pipeline = serve(manifest=manifest)
        pipeline.result = SimpleNamespace(
            frames=[[Image.new("RGB", (32, 24)) for _ in range(16)]],
            audio=np.zeros((1, 2, 8000), dtype=np.float32),
        )
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation

        # The fake pipeline takes no fps argument, so the video is encoded at the `--fps` default.
        (video,) = generation["outputs"]["video"]
        (audio,) = generation["outputs"]["audio"]
        assert {k: v for k, v in video.items() if k != "url"} == {
            "media_type": "video/mp4",
            "width": 32,
            "height": 24,
            "duration": 2.0,
            "fps": 8,
        }
        assert {k: v for k, v in audio.items() if k != "url"} == {
            "media_type": "audio/wav",
            "duration": 0.5,
            "sample_rate": 16000,
        }
        assert client.get(audio["url"]).content[:4] == b"RIFF"

    def test_validation_error_envelope(self, serve):
        client, _ = serve()
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "parameters": {"cfg_scale": 3}}
        response = client.post("/v1/generations", json=body)
        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "code": "invalid_request",
                "message": "task 'text_to_image' declares no parameter 'cfg_scale'",
                "pointer": "/parameters/cfg_scale",
            }
        }

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

    def test_health(self, serve):
        client, _ = serve()
        response = client.get("/health")
        assert (response.status_code, response.json()) == (200, {"status": "ok"})

    def test_request_id_is_generated_or_echoed(self, serve):
        client, _ = serve()
        assert client.get("/health").headers["x-request-id"], "a request without an id should be given one"
        echoed = client.get("/v1/models/unknown", headers={"x-request-id": "req-1"})
        assert (echoed.status_code, echoed.headers["x-request-id"]) == (404, "req-1")

    def test_default_seed_applies_when_request_omits_seed(self, serve):
        client, _ = serve(app_options={"default_seed": 7})
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}}
        assert client.post("/v1/generations", json=body, headers=WAIT).json()["seed"] == 7
        body["parameters"] = {"seed": 11}
        assert client.post("/v1/generations", json=body, headers=WAIT).json()["seed"] == 11

    def test_cors_is_opt_in(self, serve):
        origin = {"Origin": "http://example.com"}
        client, _ = serve()
        assert "access-control-allow-origin" not in client.get("/health", headers=origin).headers
        client, _ = serve(app_options={"enable_cors": True})
        assert client.get("/health", headers=origin).headers["access-control-allow-origin"] == "*"

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

    def test_adapters_not_advertised_when_compiled(self, serve):
        client, _ = serve(compile="{}")
        discovery = client.get(f"/v1/models/{MODEL_ID}").json()
        assert "adapters" not in discovery["server"]["capabilities"]
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "adapters": [{"type": "lora", "path": "hf:a/b"}]}
        response = client.post("/v1/generations", json=body)
        assert (response.status_code, response.json()["error"]["code"]) == (501, "not_implemented")

    def test_discovery_serves_manifest_bytes_verbatim(self, serve):
        manifest = derive_manifest(MODEL_ID, FakePipeline, FAKE_SIGNATURE, backend_kwargs("diffusers", FAKE_SIGNATURE))
        raw = json.dumps(manifest, indent=3).encode()
        client, _ = serve(manifest=raw)

        response = client.get(f"/v1/models/{MODEL_ID}")
        assert raw in response.content
        discovery = response.json()
        assert discovery["spec"] == "gms/0.1"
        assert discovery["manifest"] == manifest
        assert discovery["server"]["tasks"] == {"text_to_image": {}, "image_to_image": {}}
        assert discovery["server"]["adapters"] == {"max": 4, "sources": ["hf"]}
        assert client.get("/v1/models").json() == {
            "data": [
                {
                    "id": MODEL_ID,
                    "modalities": {"in": ["image"], "out": ["image"]},
                    "tasks": ["text_to_image", "image_to_image"],
                }
            ]
        }

    def test_unadvertised_capabilities_return_not_implemented(self, serve):
        client, _ = serve()
        for response in (
            client.post("/v1/uploads", json={}),
            client.get("/v1/generations/gen_x/outputs/video/stream"),
        ):
            assert (response.status_code, response.json()["error"]["code"]) == (501, "not_implemented")


class FakeEngine:
    """The image and video routes SGLang and vLLM-Omni share, recording what the backend sends."""

    def __init__(self):
        self.requests = []
        self.polls = 0

    def __call__(self, request):
        content_type = request.headers.get("content-type", "")
        self.requests.append((request.method, request.url.path, content_type.split(";")[0], request.content))
        path = request.url.path
        if path in ("/v1/images/generations", "/v1/images/edits"):
            buffer = io.BytesIO()
            Image.new("RGB", (16, 8), "green").save(buffer, format="PNG")
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]})
        if path == "/v1/videos" and request.method == "POST":
            return httpx.Response(200, json={"id": "video_1", "status": "queued"})
        if path == "/v1/videos/video_1" and request.method == "GET":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"id": "video_1", "status": "in_progress"})
            return httpx.Response(200, json={"id": "video_1", "status": "completed", "fps": 16.0, "duration_s": 5.0})
        if path == "/v1/videos/video_1/content":
            return httpx.Response(200, content=b"mp4-bytes")
        if path == "/v1/videos/video_1" and request.method == "DELETE":
            return httpx.Response(200, json={"id": "video_1", "deleted": True})
        return httpx.Response(404, text="no such route")


class FakeTextToVideoPipeline:
    def __call__(
        self,
        prompt: str = None,
        image: Image.Image = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ):
        pass


class TestModularPipeline:
    def test_generation_runs_a_modular_pipeline(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
        from fastapi.testclient import TestClient

        from diffusers.commands.serve.app import create_app

        pipeline = FakeImageBlock().init_pipeline()
        monkeypatch.setattr("diffusers.commands.serve.backends._load_pipeline", lambda args: pipeline)
        signature = blocks_signature(pipeline.blocks)
        kwarg_names = backend_kwargs("diffusers", signature)
        manifest = derive_manifest(MODEL_ID, type(pipeline), signature, kwarg_names)
        assert list(manifest["tasks"]) == ["text_to_image", "image_to_image"], manifest["tasks"].keys()
        assert manifest["tasks"]["text_to_image"]["inputs"] == {"prompt": {"type": "text", "required": True}}
        assert manifest["tasks"]["text_to_image"]["parameters"] == {
            "steps": {"type": "integer", "default": 4},
            "seed": {"type": "integer"},
        }

        raw = json.dumps(manifest).encode()
        served = ServedModel(
            Manifest(raw=raw, data=manifest),
            kwarg_names,
            backend_output_types("diffusers"),
            backend_capabilities(_args(), type(pipeline)),
            revision=None,
        )
        backend = DiffusersBackend(_args(), served)
        client = TestClient(create_app(served, GenerationQueue(backend, str(tmp_path))))

        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "parameters": {"seed": 3}}
        generation = client.post("/v1/generations", json=body, headers=WAIT).json()
        assert generation["status"] == "complete", generation
        assert generation["seed"] == 3
        (artifact,) = generation["outputs"]["image"]
        assert (artifact["media_type"], artifact["width"], artifact["height"]) == ("image/png", 32, 32)


class TestEngineBackend:
    def _generate(self, tmp_path, monkeypatch, served, body):
        monkeypatch.setattr("diffusers.commands.serve.backends._POLL_SECONDS", 0.0)
        engine = FakeEngine()
        client = httpx.Client(base_url="http://engine", transport=httpx.MockTransport(engine))
        backend = EngineBackend("vllm", served, client)
        resolved = served.resolve({"model": MODEL_ID, **body})
        generation = Generation(served.id, None, resolved, seeded=served.seeded(resolved.task))
        generation.directory = tmp_path
        return engine, backend.generate(generation)

    def test_exited_engine_is_unhealthy(self, tmp_path):
        pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
        from fastapi.testclient import TestClient

        from diffusers.commands.serve.app import create_app

        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        process.wait()
        served = _served(backend="vllm")
        backend = EngineBackend("vllm", served, httpx.Client(base_url="http://engine"), process)
        response = TestClient(create_app(served, GenerationQueue(backend, str(tmp_path)))).get("/health")
        expected = {"status": "unhealthy", "reason": "the vllm backend exited with code 3"}
        assert (response.status_code, response.json()) == (503, expected)

    def test_derived_manifest_keeps_only_engine_kwargs(self):
        manifest = derive_manifest(MODEL_ID, FakePipeline, FAKE_SIGNATURE, backend_kwargs("vllm", FAKE_SIGNATURE))
        assert "sigmas" not in manifest["tasks"]["text_to_image"]["parameters"]
        assert "steps" in manifest["tasks"]["text_to_image"]["parameters"]

    def test_text_to_image_request_translation(self, tmp_path, monkeypatch):
        body = {
            "inputs": {"prompt": "a cat", "negative_prompt": "blurry"},
            "parameters": {"width": 1024, "height": 768, "steps": 20, "num_outputs": 2, "seed": 5},
            "output_format": "image/webp",
        }
        engine, outputs = self._generate(tmp_path, monkeypatch, _served(backend="vllm"), body)

        ((method, path, content_type, content),) = engine.requests
        assert (method, path, content_type) == ("POST", "/v1/images/generations", "application/json")
        assert json.loads(content) == {
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "n": 2,
            "seed": 5,
            "size": "1024x768",
            "response_format": "b64_json",
            "output_format": "webp",
        }
        assert outputs == {"image": [{"file": "image-0.webp", "media_type": "image/webp", "width": 16, "height": 8}]}
        assert (tmp_path / "image-0.webp").is_file()

    def test_image_input_uses_multipart_edits(self, tmp_path, monkeypatch):
        image = {"type": "image", "base64": base64.b64encode(b"png-bytes").decode(), "media_type": "image/png"}
        body = {"inputs": {"prompt": "make it blue", "image": image}, "parameters": {"seed": 5}}
        engine, _ = self._generate(tmp_path, monkeypatch, _served(backend="sglang"), body)

        ((method, path, content_type, content),) = engine.requests
        assert (method, path, content_type) == ("POST", "/v1/images/edits", "multipart/form-data")
        assert b'name="image"; filename="image-0.png"' in content and b"png-bytes" in content
        assert b'name="prompt"\r\n\r\nmake it blue' in content

    def test_video_job_is_polled_downloaded_and_deleted(self, tmp_path, monkeypatch):
        served = _served(backend="vllm", pipeline_cls=FakeTextToVideoPipeline)
        body = {"task": "text_to_video", "inputs": {"prompt": "a cat"}, "parameters": {"num_frames": 33, "seed": 5}}
        engine, outputs = self._generate(tmp_path, monkeypatch, served, body)

        assert [(method, path) for method, path, _, _ in engine.requests] == [
            ("POST", "/v1/videos"),
            ("GET", "/v1/videos/video_1"),
            ("GET", "/v1/videos/video_1"),
            ("GET", "/v1/videos/video_1/content"),
            ("DELETE", "/v1/videos/video_1"),
        ]
        _, _, content_type, content = engine.requests[0]
        assert content_type == "multipart/form-data"
        assert b'name="num_frames"\r\n\r\n33' in content and b'name="size"\r\n\r\n832x480' in content
        assert outputs == {"video": [{"file": "video-0.mp4", "media_type": "video/mp4", "fps": 16.0, "duration": 5.0}]}
        assert (tmp_path / "video-0.mp4").read_bytes() == b"mp4-bytes"

    def test_audio_tasks_are_not_hosted(self):
        manifest = {
            "id": MODEL_ID,
            "tasks": {
                "text_to_audio": {"inputs": {"prompt": {"type": "text"}}, "outputs": {"audio": {"type": "audio"}}}
            },
        }
        assert _served(backend="vllm", manifest=manifest).tasks == {}


class TestGenerateCommand:
    def _run(self, monkeypatch, client, argv):
        parser = ArgumentParser()
        GenerateCommand.register_subcommand(parser.add_subparsers())
        monkeypatch.setattr("diffusers.commands.generate._POLL_SECONDS", 0.01)
        monkeypatch.setattr(GenerateCommand, "_client", lambda self: client)
        GenerateCommand(parser.parse_args(["generate", "--url", "http://testserver", *argv])).run()

    def test_generate_saves_outputs(self, serve, tmp_path, monkeypatch):
        client, pipeline = serve()
        output = tmp_path / "saved"
        argv = ["--inputs", '{"prompt": "a cat"}', "--parameters", '{"width": 64, "seed": 5}', "-o", str(output)]
        self._run(monkeypatch, client, argv)

        assert pipeline.calls[0]["prompt"] == "a cat"
        assert [path.name for path in output.iterdir()] == ["image-0.png"], list(output.iterdir())
        assert Image.open(output / "image-0.png").size == (64, 32)

    def test_local_file_is_sent_as_a_media_value(self, serve, tmp_path, monkeypatch):
        client, pipeline = serve()
        source = tmp_path / "cat.png"
        Image.new("RGB", (8, 8), "blue").save(source)
        inputs = json.dumps({"prompt": "make it grey", "image": str(source)})
        self._run(monkeypatch, client, ["--inputs", inputs, "-o", str(tmp_path / "saved")])

        assert pipeline.calls[0]["image"].size == (8, 8)

    def test_server_error_is_reported(self, serve, tmp_path, monkeypatch):
        client, _ = serve()
        argv = ["--inputs", '{"prompt": "a cat"}', "--parameters", '{"cfg": 1}', "-o", str(tmp_path)]
        with pytest.raises(SystemExit, match=r"HTTP 400: invalid_request: .*\(at /parameters/cfg\)"):
            self._run(monkeypatch, client, argv)


class TestServeCommand:
    def _parse(self, argv):
        parser = ArgumentParser()
        ServeCommand.register_subcommand(parser.add_subparsers())
        return parser.parse_args(["serve", *argv])

    def test_engine_backend_rejects_pipeline_flags(self):
        pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
        args = self._parse(["-m", "org/model", "--backend", "vllm", "--dtype", "bf16", "--vae-tiling"])
        with pytest.raises(SystemExit, match="--dtype, --vae-tiling configure the in-process pipeline"):
            ServeCommand(args).run()

    def test_context_parallel_is_rejected(self):
        pytest.importorskip("fastapi", reason="`diffusers-cli serve` needs the `diffusers[serve]` extra")
        with pytest.raises(SystemExit, match="--context-parallel is not supported"):
            ServeCommand(self._parse(["-m", "org/model", "--context-parallel"])).run()
