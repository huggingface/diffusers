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

"""`diffusers-cli serve --remote` — run the server in a Hugging Face Sandbox, reachable through the sandbox proxy."""

from __future__ import annotations

import shlex
import sys
import threading
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from huggingface_hub import Sandbox, get_token
from huggingface_hub.cli._output import out
from huggingface_hub.utils import httpx

from ...utils import logging
from ..run import _DEFAULT_REMOTE_DEPS, _DEFAULT_REMOTE_IMAGE, _build_task_kwargs, _kwargs_to_argv


logger = logging.get_logger("diffusers-cli/serve")

# The port the server binds inside the sandbox. `generate --sandbox-id` reaches it through the proxy.
SANDBOX_PORT = 8000
_SANDBOX_MANIFEST_PATH = "/tmp/diffusers-cli/gms.json"
_SERVE_DEPS = ("fastapi", "uvicorn")
# Each engine's own image carries the engine, its kernels and the CUDA toolchain they are built with.
_ENGINE_IMAGES = {"sglang": "lmsysorg/sglang:latest", "vllm": "vllm/vllm-omni:latest"}
_READY_POLL_SECONDS = 5.0

# Flags that say how the sandbox is set up, or that the sandbox server sets itself.
_LOCAL_KEYS = ("remote", "flavor", "dependencies", "namespace", "image", "idle_timeout", "func", "format")
_SANDBOX_SET_KEYS = ("host", "port", "public_url", "output_dir")


def add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run the server in a Hugging Face Sandbox instead of on the local machine.",
    )
    parser.add_argument(
        "--flavor",
        default="a10g-small",
        help="HF Sandbox hardware flavor for --remote (e.g. a10g-small, a100-large).",
    )
    parser.add_argument(
        "--dependencies",
        action="append",
        default=None,
        help="Extra pip dependencies to install in the sandbox. Repeat to add multiple.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace to create the sandbox under (defaults to the current user).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            f"Sandbox image for --remote. Defaults to {_DEFAULT_REMOTE_IMAGE!r}, or to the engine's own image "
            f"with --backend sglang ({_ENGINE_IMAGES['sglang']!r}) or vllm ({_ENGINE_IMAGES['vllm']!r})."
        ),
    )
    parser.add_argument(
        "--idle-timeout",
        default="30m",
        help="Auto-shutdown the sandbox after this much inactivity (e.g. 30m, 2h). Defaults to 30m.",
    )


def _announce_when_ready(sbx: Any, args: Namespace, stopped: threading.Event) -> None:
    health_url = sbx.proxy_url_for(SANDBOX_PORT, "/health")
    while not stopped.wait(_READY_POLL_SECONDS):
        try:
            ready = httpx.get(health_url, headers=sbx.proxy_headers, timeout=10.0).status_code == 200
        except httpx.HTTPError:
            ready = False
        if not ready:
            continue
        out.result(
            "remote-serve",
            sandbox_id=sbx.id,
            url=sbx.proxy_url_for(SANDBOX_PORT),
            model=args.model,
            hint=f'diffusers-cli generate --sandbox-id {sbx.id} --inputs \'{{"prompt": "..."}}\'',
        )
        return


def serve_remote(args: Namespace, task: str) -> None:
    if Path(args.model).exists():
        raise SystemExit(
            f"--model {args.model!r} is a local path; the sandbox can't see it. "
            "Pass a Hub repo id so the sandbox can download it."
        )
    if args.manifest is not None and not Path(args.manifest).is_file():
        raise SystemExit(f"--manifest {args.manifest!r} is not a file.")

    hf_token = args.token or get_token()
    logger.info(f"creating sandbox on flavor={args.flavor!r}...")
    create_kwargs: dict[str, Any] = {
        "image": args.image or _ENGINE_IMAGES.get(args.backend, _DEFAULT_REMOTE_IMAGE),
        "flavor": args.flavor,
        "forward_hf_token": True,
        "token": hf_token,
        "env": {"HF_ENABLE_PARALLEL_LOADING": "1"},
        "idle_timeout": args.idle_timeout,
        # The 120s default expires while a cold node pulls the multi-GB pytorch base image or
        # waits for GPU capacity, which fails the launch before it starts.
        "start_timeout": 600.0,
    }
    if args.namespace is not None:
        create_kwargs["namespace"] = args.namespace
    sbx = Sandbox.create(**create_kwargs)

    def _stream(chunk: str) -> None:
        sys.stderr.write(chunk)
        sys.stderr.flush()

    stopped = threading.Event()
    try:
        dependencies = [*_DEFAULT_REMOTE_DEPS, *_SERVE_DEPS, *(args.dependencies or [])]
        install_cmd = shlex.join(["uv", "pip", "install", "--system", "--break-system-packages", *dependencies])
        logger.info("installing dependencies in the sandbox...")
        sbx.run(install_cmd, on_stdout=_stream, on_stderr=_stream)
        # An image can put another environment first on PATH, with its own older `diffusers-cli`
        # (the SGLang image does), so the server is started with the interpreter the install went into.
        python = sbx.run(["uv", "python", "find", "--system"]).stdout.strip()

        task_kwargs = _build_task_kwargs(args)
        for key in (*_LOCAL_KEYS, *_SANDBOX_SET_KEYS):
            task_kwargs.pop(key, None)
        if args.manifest is not None:
            sbx.files.upload(args.manifest, _SANDBOX_MANIFEST_PATH)
            task_kwargs["manifest"] = _SANDBOX_MANIFEST_PATH
        task_kwargs["port"] = SANDBOX_PORT
        # Clients reach the server through the proxy, so output links must be built on the proxied address.
        task_kwargs["public_url"] = sbx.proxy_url_for(SANDBOX_PORT)
        cli_module = [python, "-m", "diffusers.commands.diffusers_cli"]
        cli_argv = [*cli_module, "--format", "quiet", *_kwargs_to_argv(task, task_kwargs)]

        threading.Thread(target=_announce_when_ready, args=(sbx, args, stopped), daemon=True).start()
        logger.info(f"starting the server in sandbox {sbx.id}; stop it with Ctrl-C.")
        result = sbx.run(cli_argv, on_stdout=_stream, on_stderr=_stream, check=False, capture_output=False)
    finally:
        stopped.set()
        sbx.kill()

    if result.exit_code != 0:
        raise SystemExit(f"remote server exited with code {result.exit_code}")
