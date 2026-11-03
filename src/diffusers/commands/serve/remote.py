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

"""`diffusers-cli serve --remote` — run the server in a Hugging Face Job, reachable on the port the Job exposes."""

from __future__ import annotations

import shlex
import sys
import threading
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token
from huggingface_hub.cli._output import out
from huggingface_hub.utils import httpx

from ...utils import logging
from ...utils.constants import DIFFUSERS_TRUST_REMOTE_KERNELS
from ..run import _DEFAULT_REMOTE_DEPS, _DEFAULT_REMOTE_IMAGE, _build_task_kwargs, _kwargs_to_argv


logger = logging.get_logger("diffusers-cli/serve")

# The port the server binds inside the Job. The Jobs proxy publishes it at `https://<job id>--<port>.hf.jobs`.
JOB_PORT = 8000
_SERVE_DEPS = ("fastapi", "uvicorn")
_DEFAULT_REMOTE_IDLE_TIMEOUT = "30m"
# Each engine's own image carries the engine, its kernels and the CUDA toolchain they are built with.
_ENGINE_IMAGES = {"sglang": "lmsysorg/sglang:latest", "vllm": "vllm/vllm-omni:latest"}
_READY_POLL_SECONDS = 5.0

# Flags the Job command sets itself. `_build_task_kwargs` already drops the flags that say how the Job is set up.
_JOB_SET_KEYS = ("host", "port", "public_url", "output_dir")


def add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run the server in a Hugging Face Job instead of on the local machine.",
    )
    parser.add_argument(
        "--flavor",
        default="a10g-small",
        help="HF Jobs hardware flavor for --remote (e.g. a10g-small, a100-large).",
    )
    parser.add_argument(
        "--dependencies",
        action="append",
        default=None,
        help="Extra pip dependencies to install in the Job. Repeat to add multiple.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace to run the Job under (defaults to the current user).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            f"Job image for --remote. Defaults to {_DEFAULT_REMOTE_IMAGE!r}, or to the engine's own image "
            f"with --backend sglang ({_ENGINE_IMAGES['sglang']!r}) or vllm ({_ENGINE_IMAGES['vllm']!r})."
        ),
    )
    parser.add_argument(
        "--timeout",
        default="24h",
        help=(
            "Hard limit on how long the Job runs with --remote, busy or not (e.g. 2h, 1d). A backstop for "
            "--idle-timeout. Defaults to 24h."
        ),
    )


def _announce_when_ready(api: HfApi, job: Any, args: Namespace, stopped: threading.Event) -> None:
    while not stopped.wait(_READY_POLL_SECONDS):
        urls = api.inspect_job(job_id=job.id, namespace=job.owner.name).status.expose_urls
        if not urls:
            continue
        try:
            response = httpx.get(f"{urls[0]}/health", headers={"Authorization": f"Bearer {api.token}"}, timeout=10.0)
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        out.result(
            "remote-serve",
            job_id=job.id,
            url=urls[0],
            model=args.model,
            # The HF token is left as a variable so it is not printed to the terminal.
            hint=f'curl {urls[0]}/v1/models -H "Authorization: Bearer $HF_TOKEN"',
        )
        return


def _job_script(args: Namespace, task: str) -> str:
    """The shell command the Job runs: install the dependencies, then start the server in the foreground."""
    task_kwargs = _build_task_kwargs(args)
    for key in _JOB_SET_KEYS:
        task_kwargs.pop(key, None)
    task_kwargs["host"] = "0.0.0.0"
    task_kwargs["port"] = JOB_PORT
    task_kwargs["idle_timeout"] = args.idle_timeout

    dependencies = [*_DEFAULT_REMOTE_DEPS, *_SERVE_DEPS, *(args.dependencies or [])]
    install = shlex.join(["uv", "pip", "install", "--system", "--break-system-packages", *dependencies])
    # An image can put another environment first on PATH, with its own older `diffusers-cli` (the SGLang image
    # does), so the server is started with the interpreter the install went into. The Job id is only known
    # inside the Job, and output links must be built on the public address.
    start = (
        'exec "$(uv python find --system)" -m diffusers.commands.diffusers_cli --format quiet '
        f"{shlex.join(_kwargs_to_argv(task, task_kwargs))} "
        f'--public-url "https://${{JOB_ID}}--{JOB_PORT}.hf.jobs"'
    )
    return f"{install} && {start}"


def serve_remote(args: Namespace, task: str) -> None:
    if Path(args.model).exists():
        raise SystemExit(
            f"--model {args.model!r} is a local path; the Job can't see it. "
            "Pass a Hub repo id so the Job can download it."
        )
    if args.manifest is not None:
        raise SystemExit(
            f"--manifest is not supported with --remote: the Job can't read {args.manifest!r}. Upload it as "
            f"`gms.json` to the model repo, e.g. `hf upload {args.model} {args.manifest} gms.json`, and the "
            "remote server will use it."
        )

    if args.idle_timeout is None:
        args.idle_timeout = _DEFAULT_REMOTE_IDLE_TIMEOUT
    hf_token = args.token or get_token()
    env = {"HF_ENABLE_PARALLEL_LOADING": "1"}
    if DIFFUSERS_TRUST_REMOTE_KERNELS:
        env["DIFFUSERS_TRUST_REMOTE_KERNELS"] = "true"

    api = HfApi(token=hf_token)
    logger.info(f"starting a Job on flavor={args.flavor!r}...")
    job = api.run_job(
        image=args.image or _ENGINE_IMAGES.get(args.backend, _DEFAULT_REMOTE_IMAGE),
        command=["sh", "-c", _job_script(args, task)],
        env=env,
        secrets={"HF_TOKEN": hf_token} if hf_token else None,
        flavor=args.flavor,
        timeout=args.timeout,
        expose=[JOB_PORT],
        namespace=args.namespace,
    )

    stage = None
    stopped = threading.Event()
    try:
        threading.Thread(target=_announce_when_ready, args=(api, job, args, stopped), daemon=True).start()
        logger.info(f"streaming the logs of Job {job.id}; stop it with Ctrl-C.")
        for line in api.fetch_job_logs(job_id=job.id, namespace=job.owner.name, follow=True):
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        stage = api.inspect_job(job_id=job.id, namespace=job.owner.name).status.stage
    finally:
        stopped.set()
        if stage is None:
            api.cancel_job(job_id=job.id, namespace=job.owner.name)

    if stage != "COMPLETED":
        raise SystemExit(f"remote server Job {job.id} ended with status {stage}")
