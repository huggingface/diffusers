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
import inspect
import io
import json
import re
import shlex
import threading
import time
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.utils import httpx
from PIL import Image

from diffusers.commands.run import _DEFAULT_REMOTE_IMAGE
from diffusers.commands.serve import ServeCommand, _shut_down_when_idle
from diffusers.commands.serve.backends import DiffusersBackend, _load_media, backend_capabilities
from diffusers.commands.serve.manifest import GMSError, Manifest, blocks_signature, derive_manifest
from diffusers.commands.serve.scheduler import GenerationJob, Scheduler
from diffusers.modular_pipelines import InputParam, LoopSequentialPipelineBlocks, ModularPipelineBlocks, OutputParam


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
        size = (width or 32, height or 32)
        return SimpleNamespace(images=[Image.new("RGB", size, "red") for _ in range(num_images_per_prompt)])

    def load_lora_weights(self, repo, adapter_name=None):
        self.lora_calls.append(("load", repo, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.lora_calls.append(("set", names, adapter_weights))

    def delete_adapters(self, names):
        self.lora_calls.append(("delete", names))


FAKE_SIGNATURE = inspect.signature(FakePipeline.__call__)


class FakeDenoiser(ModularPipelineBlocks):
    @property
    def inputs(self):
        return []

    def __call__(self, components, block_state, i, t):
        return components, block_state


class FakeDenoiseLoop(LoopSequentialPipelineBlocks):
    """A modular pipeline whose denoising loop, like the built-in ones, has no step callback of its own."""

    model_name = "fake"
    block_classes = [FakeDenoiser]
    block_names = ["denoiser"]

    @property
    def description(self):
        return "Denoises for `num_inference_steps` steps."

    @property
    def loop_inputs(self):
        return [InputParam("prompt"), InputParam("num_inference_steps", default=3)]

    @property
    def loop_intermediate_outputs(self):
        return [OutputParam("images")]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.timesteps = list(range(block_state.num_inference_steps))
        for i, t in enumerate(block_state.timesteps):
            components, block_state = self.loop_step(components, block_state, i=i, t=t)
        block_state.images = [Image.new("RGB", (32, 32), "red")]
        self.set_block_state(state, block_state)
        return components, state


def _args(**overrides):
    values = {
        "compile": None,
        "lora": None,
        "allowed_media_domains": None,
        "max_media_size_mb": 256,
    }
    return Namespace(**{**values, **overrides})


def _manifest(manifest=None, pipeline_cls=FakePipeline, signature=None, args=None):
    """A `Manifest` bound to `pipeline_cls`, from the given manifest dict or bytes, or derived from the pipeline."""
    signature = signature or inspect.signature(pipeline_cls.__call__)
    capabilities = backend_capabilities(args or _args(), pipeline_cls)
    if manifest is None:
        manifest = derive_manifest(MODEL_ID, pipeline_cls, signature, capabilities)
    raw = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    return Manifest(raw, json.loads(raw), pipeline_cls, signature, capabilities, max_adapters=4, revision="abc123")


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
        app = create_app(bound, Scheduler([backend], str(tmp_path)), max_wait=300.0, **(app_options or {}))
        return TestClient(app), pipeline

    return _build


VIDEO_MANIFEST = {
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
                "num_frames": {"type": "integer", "default": 121, "min": 9, "max": 257},
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
        # Built-in aliases replace the pipeline's own names; tensors, callbacks and output plumbing are dropped.
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
            ({"inputs": {}}, "invalid_request", "/inputs/prompt"),
            ({"task": "image_to_video"}, "invalid_request", "/task"),
            # `negative_prompt` is only honored while guidance_scale > 1.
            (
                {"inputs": {"prompt": "a cat", "negative_prompt": "blurry"}, "preset": "distilled"},
                "invalid_request",
                "/inputs/negative_prompt",
            ),
        ],
    )
    def test_resolve_rejects_invalid_requests(self, body, code, pointer):
        bound = _manifest(manifest=VIDEO_MANIFEST, pipeline_cls=FakeVideoPipeline)
        with pytest.raises(GMSError) as error:
            bound.resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}, **body})
        assert (error.value.code, error.value.pointer) == (code, pointer), error.value.message

    @pytest.mark.parametrize(
        "path",
        ["hf:../loras/x", "/etc/passwd", "https://x/y.safetensors"],
    )
    def test_resolve_rejects_adapter_paths_that_are_not_repo_ids(self, path):
        body = {"model": MODEL_ID, "inputs": {"prompt": "a cat"}, "adapters": [{"type": "lora", "path": path}]}
        with pytest.raises(GMSError) as error:
            _manifest().resolve(body)
        assert (error.value.code, error.value.pointer) == ("invalid_request", "/adapters/0/path"), error.value.message

    def test_resolve_maps_request_to_pipeline_kwargs(self):
        bound = _manifest(manifest=VIDEO_MANIFEST, pipeline_cls=FakeVideoPipeline)
        resolved = bound.resolve(
            {
                "model": MODEL_ID,
                "preset": "distilled",
                "inputs": {"prompt": "a cat"},
                "parameters": {"num_frames": 193, "seed": 7},
            }
        )
        # The preset overrides the task defaults.
        assert resolved.kwargs == {
            "prompt": "a cat",
            "num_frames": 193,
            "width": 768,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
        }, resolved.kwargs
        assert resolved.seed == 7
        assert resolved.task == "text_to_video"

    @pytest.mark.parametrize(
        "change, message",
        [
            # flux2-klein's published manifest declares `guidance_type`, which its pipeline does not take.
            (
                {"parameters": {"guidance_type": {"type": "string", "enum": ["distilled"]}}},
                "declares parameter 'guidance_type' on task 'text_to_video', which FakeVideoPipeline does not take",
            ),
        ],
    )
    def test_manifest_the_server_cannot_honor_fails_startup(self, change, message):
        manifest = json.loads(json.dumps(VIDEO_MANIFEST))
        if "parameters" in change:
            manifest["tasks"]["text_to_video"]["parameters"].update(change["parameters"])
        else:
            manifest.update(change)
        with pytest.raises(SystemExit, match=re.escape(message)):
            _manifest(manifest=manifest, pipeline_cls=FakeVideoPipeline)


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
        assert 'data: {"step": 3, "total_steps": 3}' in events
        assert events.rstrip().split("\n\n")[-1].startswith("id: 4\nevent: complete\n")
        # Resuming after the last progress event replays only what followed it.
        resumed = client.get(f"/v1/generations/{generation['id']}/events", headers={"Last-Event-ID": "3"}).text
        assert "event: progress" not in resumed and "event: complete" in resumed, resumed

        assert client.delete(f"/v1/generations/{generation['id']}").status_code == 204
        missing = client.get(f"/v1/generations/{generation['id']}")
        assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")
        assert client.get(artifact["url"]).status_code == 404

    def test_media_urls_are_capped_and_not_redirected(self, monkeypatch):
        def respond(request):
            if request.url.path == "/redirect":
                return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
            return httpx.Response(200, content=b"x" * 16)

        client = httpx.Client(transport=httpx.MockTransport(respond))
        monkeypatch.setattr("diffusers.commands.serve.backends.httpx.stream", client.stream)
        public = [(None, None, None, "", ("93.184.215.14", 443))]
        monkeypatch.setattr("diffusers.commands.serve.backends.socket.getaddrinfo", lambda host, port: public)
        for path, message in (("/redirect", "Redirect response"), ("/large", "larger than")):
            with pytest.raises(GMSError, match=message) as error:
                url = f"https://example.com{path}"
                _load_media("image", {"type": "image", "url": url}, "/inputs/image", None, max_bytes=8)
            assert error.value.pointer == "/inputs/image", f"{path} should be rejected as a bad input"

    @pytest.mark.parametrize(
        "address, allowed_domains, message",
        [
            ("169.254.169.254", None, "non-public address"),
            ("127.0.0.1", None, "non-public address"),
            ("93.184.215.14", ["upload.wikimedia.org"], "not in --allowed-media-domains"),
        ],
    )
    def test_media_urls_are_limited_to_public_or_allowed_hosts(self, monkeypatch, address, allowed_domains, message):
        resolved = [(None, None, None, "", (address, 80))]
        monkeypatch.setattr("diffusers.commands.serve.backends.socket.getaddrinfo", lambda host, port: resolved)
        value = {"type": "image", "url": "http://media.example/cat.png"}
        with pytest.raises(GMSError, match=message):
            _load_media("image", value, "/inputs/image", allowed_domains, max_bytes=2**20)

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
            ("delete", ["request_0", "request_1"]),
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
    def test_modular_pipeline_reports_steps_and_stops_on_cancel(self, monkeypatch, tmp_path):
        pipeline = FakeDenoiseLoop().init_pipeline()
        monkeypatch.setattr("diffusers.commands.serve.backends._load_pipeline", lambda args: pipeline)
        manifest = _manifest(pipeline_cls=type(pipeline), signature=blocks_signature(pipeline.blocks))
        backend = DiffusersBackend(_args(), manifest)

        def generate(cancel):
            generation = GenerationJob(manifest.resolve({"model": MODEL_ID, "inputs": {"prompt": "a cat"}}))
            generation.directory = tmp_path
            generation.canceled = cancel
            return backend.generate(generation), [event["step"] for _, event in generation.events]

        outputs, steps = generate(cancel=False)
        assert steps == [1, 2, 3], "every denoising step of a modular pipeline should be reported"
        assert outputs["image"], outputs
        assert generate(cancel=True) == ({}, [1]), "a canceled modular pipeline should stop at its next step"
        assert generate(cancel=False)[1] == [1, 2, 3], "the pipeline should stay usable after a cancel"


class FakeJobsApi:
    def __init__(self, logs=("server started",)):
        self.logs = logs
        self.run_kwargs = None
        self.cancelled = False
        self.telemetry = []

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
        monkeypatch.setattr(
            "diffusers.commands.serve.remote.send_telemetry", lambda **kwargs: api.telemetry.append(kwargs["topic"])
        )
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
        assert api.telemetry == ["diffusers/cli/serve/remote"], api.telemetry


class BlockingBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

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
        """Run the watcher with a 1m timeout, checked every 30s, until it shuts the server down; return when it did."""
        clock = FakeClock()
        monkeypatch.setattr("diffusers.commands.serve.time", clock)
        busy = iter(busy_checks)
        server = SimpleNamespace(should_exit=False)
        app = SimpleNamespace(state=SimpleNamespace(last_request=0.0))
        _shut_down_when_idle(server, app, SimpleNamespace(busy=lambda: next(busy, False)), 60.0)
        assert server.should_exit, "the watcher should only return once it told the server to exit"
        return clock.now

    def test_busy_server_stays_up_until_the_work_ends(self, monkeypatch):
        # Busy at the checks at 30s and 60s, so the idle clock restarts at 60s.
        assert self._watch(monkeypatch, [True, True]) == 120, "the server should exit one timeout after the work ends"
