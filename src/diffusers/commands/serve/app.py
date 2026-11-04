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

"""The Generative Media Spec HTTP surface, as a FastAPI app over one model's manifest and its scheduler."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException

from ...utils import logging
from .manifest import GMSError, Manifest
from .scheduler import GenerationJob, Scheduler


logger = logging.get_logger("diffusers-cli/serve")

MAX_WAIT_SECONDS = 300.0
REQUEST_ID_HEADER = "x-request-id"
_KEEP_ALIVE_SECONDS = 15.0
_FILE_ROUTE = "/v1/generations/{generation_id}/files/{filename}"


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_response(job: GenerationJob, base_url: str) -> dict[str, Any]:
    """The response body for a generation: outputs are linked under `base_url`, or inlined as base64.

    Raises the generation's error once it failed, which the error handler turns into the error envelope.
    """
    if job.error is not None:
        raise job.error

    outputs = None
    if job.status == "complete":
        outputs = {}
        for name, artifacts in job.outputs.items():
            outputs[name] = []
            for artifact in artifacts:
                entry = {key: value for key, value in artifact.items() if key != "file"}
                if job.request.response_format == "b64_json":
                    entry["base64"] = base64.b64encode((job.directory / artifact["file"]).read_bytes()).decode()
                else:
                    entry["url"] = base_url + _FILE_ROUTE.format(generation_id=job.id, filename=artifact["file"])
                outputs[name].append(entry)

    return {
        "id": job.id,
        "model": job.model,
        "revision": job.revision,
        "status": job.status,
        "created_at": _timestamp(job.created_at),
        "progress": job.progress if job.status == "generating" else None,
        "outputs": outputs,
        "expires_at": _timestamp(job.expires_at),
        "seed": job.seed if job.status == "complete" else None,
        "usage": job.usage,
    }


def create_app(
    manifest: Manifest,
    scheduler: Scheduler,
    default_seed: int | None = None,
    enable_cors: bool = False,
    public_url: str | None = None,
) -> FastAPI:
    # A proxy may or may not strip the path it publishes the server under before forwarding a request.
    # With that path as the root path, routing accepts both forms.
    root_path = urlparse(public_url).path.rstrip("/") if public_url is not None else ""
    app = FastAPI(title="diffusers-cli serve", docs_url=None, redoc_url=None, openapi_url=None, root_path=root_path)
    app.state.last_request = time.monotonic()

    def base_url(request: Request) -> str:
        # Output links must work for the client, so they are built on the address it reached the server at.
        return (public_url or str(request.base_url)).rstrip("/")

    @app.middleware("http")
    async def request_id(request: Request, call_next) -> Response:
        request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    @app.middleware("http")
    async def track_activity(request: Request, call_next) -> Response:
        # Health checks come from monitors rather than users, and must not keep an idle server alive.
        if not request.url.path.rstrip("/").endswith("/health"):
            app.state.last_request = time.monotonic()
        return await call_next(request)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER],
        )
        logger.warning("--enable-cors allows requests from any origin. Not recommended for production.")

    @app.exception_handler(GMSError)
    async def gms_error(request: Request, error: GMSError) -> JSONResponse:
        return JSONResponse(error.to_dict(), status_code=error.status)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            code = "not_found"
        elif error.status_code < 500:
            code = "invalid_request"
        else:
            code = "internal"
        return JSONResponse({"error": {"code": code, "message": str(error.detail)}}, status_code=error.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        message = f"{type(error).__name__}: {error}"
        # An unhandled error is answered outside the middleware stack, so the request id is attached here.
        headers = {REQUEST_ID_HEADER: request.state.request_id}
        return JSONResponse({"error": {"code": "internal", "message": message}}, status_code=500, headers=headers)

    @app.get("/health")
    async def health() -> JSONResponse:
        failure = scheduler.failure()
        if failure is not None:
            return JSONResponse({"status": "unhealthy", "reason": failure}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"data": [manifest.summary()]}

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> Response:
        if model_id != manifest.id:
            raise GMSError("not_found", f"model {model_id!r} is not served here; this server hosts {manifest.id!r}")
        return Response(manifest.discovery_document(), media_type="application/json")

    @app.post("/v1/generations", status_code=201)
    async def create_generation(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise GMSError("invalid_request", f"request body is not valid JSON: {e}") from e
        resolved = manifest.resolve(body)
        seeded = manifest.seeded(resolved.task)
        generation = GenerationJob(manifest.id, manifest.revision, resolved, seeded=seeded, default_seed=default_seed)
        scheduler.submit(generation)

        wait = re.fullmatch(r"\s*wait=(\d+)\s*", request.headers.get("prefer", ""))
        # `Prefer` is a hint, so a server without the `wait` capability answers right away instead of failing.
        if wait is not None and "wait" in manifest.capabilities:
            await asyncio.to_thread(generation.wait, min(float(wait.group(1)), MAX_WAIT_SECONDS))
        return _job_response(generation, base_url(request))

    @app.get("/v1/generations/{generation_id}")
    async def get_generation(generation_id: str, request: Request) -> dict[str, Any]:
        generation = scheduler.get(generation_id)
        return _job_response(generation, base_url(request))

    @app.delete("/v1/generations/{generation_id}", status_code=204)
    async def delete_generation(generation_id: str) -> Response:
        scheduler.delete(generation_id)
        return Response(status_code=204)

    @app.post("/v1/generations/{generation_id}/cancel")
    async def cancel_generation(generation_id: str, request: Request) -> dict[str, Any]:
        generation = scheduler.cancel(generation_id)
        return _job_response(generation, base_url(request))

    @app.get("/v1/generations/{generation_id}/events")
    async def get_generation_events(generation_id: str, request: Request) -> StreamingResponse:
        if "sse" not in manifest.capabilities:
            raise GMSError("not_implemented", "this server does not advertise the `sse` capability")
        generation = scheduler.get(generation_id)
        last_event_id = request.headers.get("last-event-id", "")
        start = int(last_event_id) if last_event_id.isdigit() and len(last_event_id) < 10 else 0

        async def stream():
            index = start
            while True:
                events, done = await asyncio.to_thread(generation.wait_for_events, index, _KEEP_ALIVE_SECONDS)
                for event, data in events:
                    index += 1
                    payload = _job_response(generation, base_url(request)) if event == "complete" else data
                    yield f"id: {index}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"
                if done:
                    return
                if not events:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get(_FILE_ROUTE)
    async def get_generation_file(generation_id: str, filename: str) -> FileResponse:
        generation = scheduler.get(generation_id)
        artifacts = [a for group in (generation.outputs or {}).values() for a in group if a["file"] == filename]
        if not artifacts:
            raise GMSError("not_found", f"generation {generation_id!r} has no file {filename!r}")
        return FileResponse(generation.directory / filename, media_type=artifacts[0]["media_type"])

    @app.get("/v1/generations/{generation_id}/outputs/{name}/stream")
    async def stream_generation_output(generation_id: str, name: str) -> Response:
        raise GMSError("not_implemented", "this server does not advertise the `stream` capability")

    @app.post("/v1/uploads")
    async def create_upload() -> Response:
        raise GMSError("not_implemented", "this server does not advertise the `presigned_upload` capability")

    return app
