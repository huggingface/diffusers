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

"""`diffusers-cli serve` — serve a pipeline over HTTP with the Generative Media Spec (GMS) API.

Generations run one at a time on a background worker: a request returns immediately with a queued generation that
clients poll or follow over server-sent events.
"""

from __future__ import annotations

import inspect
import random
import secrets
import threading
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter, _SubParsersAction
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.cli._output import out

from ...utils import logging
from .. import BaseDiffusersCLICommand
from ..run import _add_loading_arguments, _add_optimization_arguments, _duration_to_seconds
from .remote import add_remote_arguments, serve_remote


logger = logging.get_logger("diffusers-cli/serve")

DEFAULT_OUTPUT_DIR = str(Path.home() / ".diffusers" / "cli" / "serve" / "outputs")


def _resolve_pipeline(args: Namespace) -> tuple[type, inspect.Signature, str | None]:
    """Return the pipeline class, the signature of its call, and the commit hash of a standard Hub repo."""
    import diffusers

    from .manifest import blocks_signature

    # A modular repo ships `modular_model_index.json` instead of `model_index.json`, so `load_config` raises.
    try:
        config, commit_hash = diffusers.DiffusionPipeline.load_config(
            args.model, token=args.token, revision=args.revision, return_commit_hash=True
        )
    except OSError:
        config, commit_hash = None, None

    if config is not None:
        class_name = config.get("_class_name")
        pipeline_cls = getattr(diffusers, str(class_name), None)
        if not isinstance(pipeline_cls, type):
            raise SystemExit(
                f"Pipeline class {class_name!r} declared in {diffusers.DiffusionPipeline.config_name} is not "
                "exported by the installed diffusers."
            )
        if not issubclass(pipeline_cls, diffusers.ModularPipeline):
            return pipeline_cls, inspect.signature(pipeline_cls.__call__), commit_hash

    try:
        pipeline = diffusers.ModularPipeline.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, token=args.token, revision=args.revision
        )
    except OSError as e:
        raise SystemExit(f"Could not read a pipeline config for {args.model!r}: {e}") from e
    return type(pipeline), blocks_signature(pipeline.blocks), commit_hash


def _shut_down_when_idle(server: Any, app: Any, scheduler: Any, idle_timeout: str) -> None:
    limit = _duration_to_seconds(idle_timeout)
    while True:
        time.sleep(min(limit, 30.0))
        if scheduler.busy():
            app.state.last_request = time.monotonic()
            continue
        if time.monotonic() - app.state.last_request >= limit:
            logger.info(f"No requests for {idle_timeout}; shutting down.")
            server.should_exit = True
            return


class ServeCommand(BaseDiffusersCLICommand):
    task = "serve"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        epilog = (
            "Examples\n"
            "  $ diffusers-cli serve black-forest-labs/FLUX.1-dev\n"
            "  $ diffusers-cli serve black-forest-labs/FLUX.1-dev --compile --port 9000\n"
            "\n"
            "  $ curl localhost:8000/v1/generations -H 'Prefer: wait=120' \\\n"
            '      -d \'{"model": "black-forest-labs/FLUX.1-dev", "inputs": {"prompt": "a cat on the moon"}}\'\n'
            "\n"
            "Learn more\n"
            "  Use `diffusers-cli <command> --help` for more information about a command.\n"
            "  Read the documentation at https://huggingface.co/docs/diffusers\n"
        )

        parser: ArgumentParser = subparsers.add_parser(
            "serve",
            help="Serve a diffusers pipeline over HTTP with the Generative Media Spec API.",
            usage="\n  diffusers-cli serve <model> [options]",
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter,
        )
        parser._optionals.title = "Options"
        parser.add_argument(
            "model_id",
            nargs="?",
            default=None,
            metavar="MODEL",
            help="Model id on the Hugging Face Hub or local path.",
        )
        _add_loading_arguments(parser, require_model=False)
        _add_optimization_arguments(parser)
        parser.add_argument("--host", default="127.0.0.1", help="Interface to bind. Defaults to 127.0.0.1.")
        parser.add_argument("--port", type=int, default=8000, help="Port to bind. Defaults to 8000.")
        parser.add_argument(
            "--public-url",
            default=None,
            metavar="URL",
            help=(
                "Base URL clients reach the server at, used to build output file links. Set it when the server "
                "sits behind a reverse proxy; defaults to the address of each incoming request."
            ),
        )
        parser.add_argument(
            "--manifest",
            default=None,
            metavar="PATH",
            help=(
                "Path to a GMS manifest (`gms.json`) describing the model. Defaults to the `gms.json` published "
                "in the model repo; when there is none, a manifest is derived from the pipeline's call signature."
            ),
        )
        parser.add_argument(
            "--output-dir",
            default=DEFAULT_OUTPUT_DIR,
            help=f"Directory generated files are stored in until they expire. Defaults to {DEFAULT_OUTPUT_DIR}.",
        )
        parser.add_argument(
            "--data-parallel",
            type=int,
            default=1,
            metavar="N",
            help=(
                "Load a copy of the pipeline on each of the first N GPUs and run up to N generations at once. "
                "Defaults to 1."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Seed for the generator that picks the seed of each request that does not set one. "
                "Defaults to a random seed chosen at startup."
            ),
        )
        parser.add_argument(
            "--enable-cors",
            action="store_true",
            help="Allow cross-origin requests from any origin, so browser apps on other hosts can call the server.",
        )
        parser.add_argument(
            "--log-level",
            choices=tuple(logging.get_log_levels_dict()),
            default=None,
            help="Logging level for diffusers and the HTTP server. Defaults to the current diffusers verbosity.",
        )
        parser.add_argument(
            "--idle-timeout",
            default=None,
            help=(
                "Shut the server down after this long without requests while no generation is queued or running "
                "(e.g. 30m, 2h). Defaults to never, or to 30m with --remote."
            ),
        )
        add_remote_arguments(parser)
        # `_load_pipeline` reads these `run` flags, which `serve` does not expose.
        # Most served models are DiTs trained in bf16; `run`'s `auto` would load them in fp32.
        parser.set_defaults(func=ServeCommand, workflow=None, offload_margin=None, dtype="bf16")

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        if args.model_id is not None and args.model is not None and args.model_id != args.model:
            raise SystemExit(f"Got two models: {args.model_id!r} and --model {args.model!r}. Pass only one.")
        args.model = args.model or args.model_id
        # Cleared so `serve --remote` does not forward it to the Job as `--model-id`.
        args.model_id = None
        if args.model is None:
            raise SystemExit("Pass the model to serve, e.g. `diffusers-cli serve black-forest-labs/FLUX.1-dev`.")
        if args.idle_timeout is not None:
            try:
                _duration_to_seconds(args.idle_timeout)
            except ValueError:
                raise SystemExit(f"--idle-timeout {args.idle_timeout!r} is not a duration such as 30m or 2h.")
        if args.log_level is not None:
            logging.set_verbosity(logging.get_log_levels_dict()[args.log_level])

        if args.context_parallel and args.device_map is not None:
            raise SystemExit(
                "--device-map does not apply with --context-parallel: each worker loads the pipeline on its own GPU."
            )
        if args.data_parallel < 1:
            raise SystemExit(f"--data-parallel must be at least 1, got {args.data_parallel}.")
        if args.data_parallel > 1 and args.context_parallel:
            raise SystemExit(
                "--data-parallel and --context-parallel cannot be combined: --context-parallel already uses "
                "every visible GPU for each generation."
            )
        if args.data_parallel > 1 and args.device_map is not None:
            raise SystemExit(
                "--device-map does not apply with --data-parallel: each copy of the pipeline is loaded on its own GPU."
            )
        if args.data_parallel > 1 and not args.remote and torch.cuda.device_count() < args.data_parallel:
            raise SystemExit(
                f"--data-parallel {args.data_parallel} needs {args.data_parallel} GPUs, but "
                f"{torch.cuda.device_count()} are visible."
            )

        if args.remote:
            serve_remote(args, self.task)
            return

        try:
            import fastapi  # noqa: F401
            import uvicorn
        except ImportError as e:
            raise SystemExit(
                '`diffusers-cli serve` requires FastAPI and uvicorn. Install them with: pip install "diffusers[serve]"'
            ) from e

        from .app import create_app
        from .backends import DiffusersBackend, ParallelBackend, backend_capabilities
        from .manifest import load_manifest
        from .scheduler import Scheduler

        pipeline_cls, signature, commit_hash = _resolve_pipeline(args)
        capabilities = backend_capabilities(args, pipeline_cls)
        manifest = load_manifest(
            args.model,
            pipeline_cls,
            signature,
            capabilities,
            manifest_path=args.manifest,
            revision=args.revision,
            commit_hash=commit_hash,
            token=args.token,
        )

        if args.context_parallel:
            backends = [ParallelBackend(args, manifest)]
        elif args.data_parallel > 1:
            backends = [
                DiffusersBackend(Namespace(**{**vars(args), "device_map": f"cuda:{index}"}), manifest)
                for index in range(args.data_parallel)
            ]
        else:
            backends = [DiffusersBackend(args, manifest)]

        try:
            seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
            # Seeds the generator `Manifest.resolve` draws request seeds from, like vLLM's `--seed`.
            random.seed(seed)
            scheduler = Scheduler(backends, args.output_dir)
            app = create_app(
                manifest,
                scheduler,
                enable_cors=args.enable_cors,
                public_url=args.public_url,
            )
            out.result(
                self.task,
                url=f"http://{args.host}:{args.port}",
                model=manifest.id,
                pipeline_class=pipeline_cls.__name__,
                tasks=list(manifest.tasks),
                capabilities=manifest.capabilities,
                seed=seed,
            )
            server = uvicorn.Server(
                uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level or "info")
            )
            if args.idle_timeout is not None:
                threading.Thread(
                    target=_shut_down_when_idle,
                    args=(server, app, scheduler, args.idle_timeout),
                    name="diffusers-serve-idle",
                    daemon=True,
                ).start()
            server.run()
        finally:
            for backend in backends:
                backend.close()
