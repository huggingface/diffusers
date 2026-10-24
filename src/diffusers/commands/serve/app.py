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

"""The Generative Media Spec HTTP surface, as a FastAPI app over one served model and one generation queue."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException

from ...utils import logging
from .generations import Generation, GenerationQueue
from .manifest import GMSError, ServedModel


logger = logging.get_logger("diffusers-cli/serve")

MAX_WAIT_SECONDS = 300.0
REQUEST_ID_HEADER = "x-request-id"
_KEEP_ALIVE_SECONDS = 15.0


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _render(generation: Generation, request: Request, public_url: str | None) -> dict[str, Any]:
    outputs = None
    if generation.status == "complete":
        outputs = {}
        for name, artifacts in generation.outputs.items():
            outputs[name] = []
            for artifact in artifacts:
                rendered = {key: value for key, value in artifact.items() if key != "file"}
                if generation.request.response_format == "b64_json":
                    content = (generation.directory / artifact["file"]).read_bytes()
                    rendered["base64"] = base64.b64encode(content).decode()
                else:
                    route = {"generation_id": generation.id, "filename": artifact["file"]}
                    if public_url is None:
                        rendered["url"] = str(request.url_for("get_generation_file", **route))
                    else:
                        path = request.app.url_path_for("get_generation_file", **route)
                        rendered["url"] = public_url.rstrip("/") + path
                outputs[name].append(rendered)

    return {
        "id": generation.id,
        "model": generation.model,
        "revision": generation.revision,
        "status": generation.status,
        "created_at": _timestamp(generation.created_at),
        "progress": generation.progress if generation.status == "generating" else None,
        "outputs": outputs,
        "expires_at": _timestamp(generation.expires_at),
        "seed": generation.seed if generation.status == "complete" else None,
        "usage": generation.usage,
    }


def create_app(
    served: ServedModel,
    generations: GenerationQueue,
    default_seed: int | None = None,
    enable_cors: bool = False,
    public_url: str | None = None,
) -> FastAPI:
    # A proxy may or may not strip the path it publishes the server under before forwarding a request.
    # With that path as the root path, routing accepts both forms.
    root_path = urlparse(public_url).path.rstrip("/") if public_url is not None else ""
    app = FastAPI(title="diffusers-cli serve", docs_url=None, redoc_url=None, openapi_url=None, root_path=root_path)

    @app.middleware("http")
    async def request_id(request: Request, call_next) -> Response:
        request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

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
        failure = generations.failure()
        if failure is not None:
            return JSONResponse({"status": "unhealthy", "reason": failure}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"data": [served.summary()]}

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> Response:
        if model_id != served.id:
            raise GMSError("not_found", f"model {model_id!r} is not served here; this server hosts {served.id!r}")
        return Response(served.discovery_document(), media_type="application/json")

    @app.post("/v1/generations", status_code=201)
    async def create_generation(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise GMSError("invalid_request", f"request body is not valid JSON: {e}") from e
        resolved = served.resolve(body)
        seeded = served.seeded(resolved.task)
        generation = Generation(served.id, served.revision, resolved, seeded=seeded, default_seed=default_seed)
        generations.submit(generation)

        wait = re.fullmatch(r"\s*wait=(\d+)\s*", request.headers.get("prefer", ""))
        if wait is not None:
            await asyncio.to_thread(generation.wait, min(float(wait.group(1)), MAX_WAIT_SECONDS))
        if generation.error is not None:
            raise generation.error
        return _render(generation, request, public_url)

    @app.get("/v1/generations/{generation_id}")
    async def get_generation(generation_id: str, request: Request) -> dict[str, Any]:
        generation = generations.get(generation_id)
        if generation.error is not None:
            raise generation.error
        return _render(generation, request, public_url)

    @app.delete("/v1/generations/{generation_id}", status_code=204)
    async def delete_generation(generation_id: str) -> Response:
        generations.delete(generation_id)
        return Response(status_code=204)

    @app.post("/v1/generations/{generation_id}/cancel")
    async def cancel_generation(generation_id: str, request: Request) -> dict[str, Any]:
        generation = generations.cancel(generation_id)
        if generation.error is not None:
            raise generation.error
        return _render(generation, request, public_url)

    @app.get("/v1/generations/{generation_id}/events")
    async def get_generation_events(generation_id: str, request: Request) -> StreamingResponse:
        generation = generations.get(generation_id)
        last_event_id = request.headers.get("last-event-id", "")
        start = int(last_event_id) if last_event_id.isdigit() and len(last_event_id) < 10 else 0

        async def stream():
            index = start
            while True:
                events, done = await asyncio.to_thread(generation.wait_for_events, index, _KEEP_ALIVE_SECONDS)
                for event, data in events:
                    index += 1
                    payload = _render(generation, request, public_url) if event == "complete" else data
                    yield f"id: {index}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"
                if done:
                    return
                if not events:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/generations/{generation_id}/files/{filename}")
    async def get_generation_file(generation_id: str, filename: str) -> FileResponse:
        generation = generations.get(generation_id)
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
