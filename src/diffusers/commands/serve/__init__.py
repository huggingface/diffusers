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

Generations run on background workers, one per pipeline copy: a request returns immediately with a queued generation
that clients poll or follow over server-sent events.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import random
import secrets
import threading
import time
import typing
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter, _SubParsersAction
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.cli._output import out

from ...utils import BaseOutput, logging
from .. import BaseDiffusersCLICommand
from ..run import _add_loading_arguments, _add_optimization_arguments, _duration_to_seconds
from .remote import add_remote_arguments, serve_remote


logger = logging.get_logger("diffusers-cli/serve")

DEFAULT_OUTPUT_DIR = str(Path.home() / ".diffusers" / "cli" / "serve" / "outputs")


def _resolve_pipeline(
    args: Namespace,
) -> tuple[type, inspect.Signature, set[str] | None, str | None, dict[str, Any] | None]:
    """Return the pipeline class, the signature of its call, the fields of its output, the commit hash of a
    standard Hub repo, and the workflow map of a modular pipeline, narrowed to --workflow when it is set. The output
    fields are None when the pipeline doesn't annotate what it returns."""
    import diffusers

    from .manifest import blocks_signature

    # A modular repo ships `modular_model_index.json` instead of `model_index.json`, so `load_config` raises. A repo
    # may also ship a `model_index.json` for another framework (e.g. SGLang) next to its modular one.
    try:
        config, commit_hash = diffusers.DiffusionPipeline.load_config(
            args.model, token=args.token, revision=args.revision, return_commit_hash=True
        )
    except OSError:
        config, commit_hash = None, None

    class_name = config.get("_class_name") if config is not None else None
    pipeline_cls = getattr(diffusers, class_name, None) if class_name is not None else None
    if isinstance(pipeline_cls, type) and not issubclass(pipeline_cls, diffusers.ModularPipeline):
        # Evaluated, so a module with `from __future__ import annotations` gives types rather than strings. A kwarg
        # whose annotation doesn't evaluate is left out of a derived manifest.
        try:
            signature = inspect.signature(pipeline_cls.__call__, eval_str=True)
        except (NameError, TypeError):
            signature = inspect.signature(pipeline_cls.__call__)
        returned = typing.get_args(signature.return_annotation) or (signature.return_annotation,)
        outputs = [cls for cls in returned if isinstance(cls, type) and issubclass(cls, BaseOutput)]
        output_fields = {field.name for field in dataclasses.fields(outputs[0])} if outputs else None
        return pipeline_cls, signature, output_fields, commit_hash, None

    try:
        pipeline = diffusers.ModularPipeline.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            token=args.token,
            revision=args.revision,
        )
    # An unknown modular class raises AttributeError rather than OSError.
    except (OSError, AttributeError) as e:
        if class_name is not None and not isinstance(pipeline_cls, type):
            raise SystemExit(
                f"Pipeline class {class_name!r} declared in {diffusers.DiffusionPipeline.config_name} is not "
                "exported by the installed diffusers, and the repo has no modular pipeline to load instead."
            ) from e
        raise SystemExit(f"Could not read a pipeline config for {args.model!r}: {e}") from e
    # The blocks of one workflow no longer carry the workflow map, so its triggers are read before pruning.
    blocks = pipeline.blocks
    workflow_map = getattr(blocks, "_workflow_map", None)
    if args.workflow is not None:
        if not workflow_map or args.workflow not in workflow_map:
            raise SystemExit(
                f"--workflow {args.workflow!r} is not a workflow of {type(pipeline).__name__}. Its workflows are "
                f"{list(workflow_map or [])}."
            )
        blocks = blocks.get_workflow(args.workflow)
        workflow_map = {args.workflow: workflow_map[args.workflow]}
    output_fields = {output.name for output in blocks.intermediate_outputs}
    return type(pipeline), blocks_signature(blocks), output_fields, commit_hash, workflow_map


def _shut_down_when_idle(server: Any, app: Any, scheduler: Any, limit: float) -> None:
    while True:
        time.sleep(min(limit, 30.0))
        if scheduler.busy():
            app.state.last_request = time.monotonic()
            continue
        if time.monotonic() - app.state.last_request >= limit:
            logger.info(f"No requests for {limit:g}s; shutting down.")
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
            "--workflow",
            default=None,
            help=(
                "For modular pipelines: serve only this workflow (e.g. t2va), loading only its components. "
                "Defaults to loading every component and serving each workflow the pipeline declares as a task."
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
            "--allowed-media-domains",
            action="append",
            default=None,
            metavar="HOST",
            help=(
                "Only fetch media input URLs from this host (e.g. upload.wikimedia.org). Repeat to allow several. "
                "Defaults to any host with a public address."
            ),
        )
        parser.add_argument(
            "--max-adapters",
            type=int,
            default=4,
            metavar="N",
            help="Most adapters a request may apply. Defaults to 4.",
        )
        parser.add_argument(
            "--max-wait",
            default="5m",
            help="Longest a request may hold its response with `Prefer: wait` (e.g. 5m). Defaults to 5m.",
        )
        parser.add_argument(
            "--max-media-size-mb",
            type=int,
            default=256,
            metavar="MB",
            help="Largest media input the server downloads from a URL, in MB. Defaults to 256.",
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
            "--generation-timeout",
            default=None,
            help=(
                "Fail a generation that runs longer than this (e.g. 10m). It is interrupted at its next step; "
                "/health reports a generation still running at twice the limit. Defaults to no limit."
            ),
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
        # `_load_pipeline` reads `offload_margin`, a `run` flag that `serve` does not expose. `dtype` defaults to bf16
        # because most served models are DiTs trained in it; `run`'s `auto` would load them in fp32.
        parser.set_defaults(func=ServeCommand, offload_margin=None, dtype="bf16")

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
        seconds = {}
        for flag, value in (
            ("--idle-timeout", args.idle_timeout),
            ("--generation-timeout", args.generation_timeout),
            ("--max-wait", args.max_wait),
        ):
            if value is None:
                seconds[flag] = None
                continue
            try:
                seconds[flag] = _duration_to_seconds(value)
            except ValueError:
                raise SystemExit(f"{flag} {value!r} is not a duration such as 30m or 2h.")
            if seconds[flag] <= 0:
                raise SystemExit(f"{flag} must be a positive duration such as 30m or 2h, got {value!r}.")
        for flag, value in (("--max-adapters", args.max_adapters), ("--max-media-size-mb", args.max_media_size_mb)):
            if value < 0:
                raise SystemExit(f"{flag} must not be negative, got {value}.")
        if args.log_level is not None:
            logging.set_verbosity(logging.get_log_levels_dict()[args.log_level])

        # TODO(DN6): support --context-parallel once the encoders and the VAE decode are not repeated on every rank.
        if args.context_parallel:
            raise SystemExit(
                "--context-parallel is not supported by `diffusers-cli serve` yet. Use --data-parallel N to serve "
                "on multiple GPUs."
            )
        if args.data_parallel < 1:
            raise SystemExit(f"--data-parallel must be at least 1, got {args.data_parallel}.")
        if args.data_parallel > 1 and args.compile is not None:
            raise SystemExit(
                "--compile is not supported with --data-parallel yet: the copies of the pipeline would compile "
                "concurrently on separate threads, which torch.compile does not support."
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
        from .backends import DiffusersBackend, backend_capabilities
        from .manifest import Manifest, derive_manifest, load_manifest
        from .scheduler import Scheduler

        pipeline_cls, signature, output_fields, commit_hash, workflow_map = _resolve_pipeline(args)
        capabilities = backend_capabilities(args, pipeline_cls)
        published = load_manifest(args.model, args.manifest, revision=args.revision, token=args.token)
        if published is None:
            model_id = Path(args.model).name if Path(args.model).is_dir() else args.model
            data = derive_manifest(model_id, pipeline_cls, signature, capabilities, output_fields, workflow_map)
            raw, source = json.dumps(data).encode(), None
        else:
            raw, data, source = published
        manifest = Manifest(
            raw,
            data,
            pipeline_cls,
            signature,
            capabilities,
            source=source,
            revision=commit_hash,
            output_fields=output_fields,
            max_adapters=args.max_adapters,
        )

        if args.data_parallel > 1:
            backends = []
            for index in range(args.data_parallel):
                replica_args = copy.copy(args)
                replica_args.device_map = f"cuda:{index}"
                backends.append(DiffusersBackend(replica_args, manifest))
        else:
            backends = [DiffusersBackend(args, manifest)]

        seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
        # Seeds the generator `Manifest.resolve` draws request seeds from, like vLLM's `--seed`.
        random.seed(seed)
        scheduler = Scheduler(backends, args.output_dir, timeout=seconds["--generation-timeout"])
        app = create_app(
            manifest,
            scheduler,
            enable_cors=args.enable_cors,
            public_url=args.public_url,
            max_wait=seconds["--max-wait"],
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
                args=(server, app, scheduler, seconds["--idle-timeout"]),
                name="diffusers-serve-idle",
                daemon=True,
            ).start()
        server.run()
