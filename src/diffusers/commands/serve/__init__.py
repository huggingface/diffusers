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
clients poll or follow over server-sent events. The pipeline runs in-process by default, or behind an SGLang or
vLLM-Omni server with `--backend`.
"""

from __future__ import annotations

import inspect
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter, _SubParsersAction
from pathlib import Path

from huggingface_hub.cli._output import out

from ...utils import logging
from .. import BaseDiffusersCLICommand
from ..run import _add_loading_arguments, _add_optimization_arguments
from .remote import add_remote_arguments, serve_remote


logger = logging.get_logger("diffusers-cli/serve")

DEFAULT_OUTPUT_DIR = str(Path.home() / ".diffusers" / "cli" / "serve" / "outputs")
BACKEND_CHOICES = ("diffusers", "sglang", "vllm")

# Flags that configure the in-process pipeline, paired with the value argparse gives them when unset.
_DIFFUSERS_ONLY_FLAGS = {
    "device_map": None,
    "dtype": "auto",
    "variant": None,
    "lora": None,
    "cpu_offload": None,
    "attention_backend": "default",
    "vae_tiling": False,
    "vae_slicing": False,
    "compile": None,
    "compile_mode": "regional",
}


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

    if args.backend != "diffusers":
        raise SystemExit(
            f"{args.model!r} is a modular pipeline, which --backend {args.backend} cannot serve. "
            "Use --backend diffusers."
        )
    try:
        pipeline = diffusers.ModularPipeline.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, token=args.token, revision=args.revision
        )
    except OSError as e:
        raise SystemExit(f"Could not read a pipeline config for {args.model!r}: {e}") from e
    return type(pipeline), blocks_signature(pipeline.blocks), commit_hash


class ServeCommand(BaseDiffusersCLICommand):
    task = "serve"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        epilog = (
            "Examples\n"
            "  $ diffusers-cli serve -m black-forest-labs/FLUX.1-dev --dtype bf16\n"
            "  $ diffusers-cli serve -m black-forest-labs/FLUX.1-dev --dtype bf16 --compile --port 9000\n"
            "  $ diffusers-cli serve -m Wan-AI/Wan2.2-T2V-A14B-Diffusers --backend vllm\n"
            "  $ diffusers-cli serve -m Qwen/Qwen-Image --backend sglang --backend-args '--num-gpus 2'\n"
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
            usage="\n  diffusers-cli serve [options]",
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter,
        )
        parser._optionals.title = "Options"
        _add_loading_arguments(parser)
        _add_optimization_arguments(parser)
        parser.add_argument(
            "--backend",
            choices=BACKEND_CHOICES,
            default="diffusers",
            help=(
                "What runs the model. 'diffusers' loads the pipeline in this process. 'sglang' and 'vllm' start "
                "`sglang serve` / `vllm serve --omni` as a child process and translate requests to its "
                "OpenAI-compatible image and video routes."
            ),
        )
        parser.add_argument(
            "--backend-args",
            default=None,
            metavar="ARGS",
            help=(
                "Extra arguments appended to the `sglang serve` / `vllm serve` command, as one quoted string "
                "(e.g. '--num-gpus 2'). Only used with --backend sglang or vllm."
            ),
        )
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
            "--fps",
            type=int,
            default=8,
            help="FPS used to encode video outputs when the pipeline call takes no fps argument.",
        )
        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=None,
            help="Sample rate used to encode audio outputs. Defaults to 16000.",
        )
        parser.add_argument(
            "--default-seed",
            type=int,
            default=None,
            help="Seed used for requests that do not set one. Defaults to a random seed per request.",
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
        add_remote_arguments(parser)
        # `_load_pipeline` reads these `run` flags, which `serve` does not expose.
        parser.set_defaults(func=ServeCommand, workflow=None, offload_margin=None)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        if args.log_level is not None:
            logging.set_verbosity(logging.get_log_levels_dict()[args.log_level])

        if args.context_parallel:
            raise SystemExit(
                "--context-parallel is not supported by `diffusers-cli serve`: under torchrun every rank would "
                "start its own server on the same port."
            )
        if args.backend == "diffusers" and args.backend_args is not None:
            raise SystemExit("--backend-args only applies to --backend sglang or vllm.")
        if args.backend != "diffusers":
            ignored = [
                "--" + name.replace("_", "-")
                for name, unset in _DIFFUSERS_ONLY_FLAGS.items()
                if getattr(args, name) != unset
            ]
            if ignored:
                logger.warning(
                    f"{', '.join(ignored)} configure the in-process pipeline and are ignored with "
                    f"--backend {args.backend}. Pass the engine's own flags through --backend-args instead."
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
        from .backends import (
            DiffusersBackend,
            EngineBackend,
            backend_capabilities,
            backend_kwargs,
            backend_output_types,
        )
        from .generations import GenerationQueue
        from .manifest import ServedModel, load_manifest

        pipeline_cls, signature, commit_hash = _resolve_pipeline(args)
        kwarg_names = backend_kwargs(args.backend, signature)
        manifest = load_manifest(
            args.model,
            pipeline_cls,
            signature,
            kwarg_names,
            manifest_path=args.manifest,
            revision=args.revision,
            token=args.token,
        )
        served = ServedModel(
            manifest,
            kwarg_names,
            backend_output_types(args.backend),
            backend_capabilities(args, pipeline_cls),
            revision=commit_hash,
        )
        if not served.tasks:
            raise SystemExit(
                f"None of the tasks {list(manifest.data['tasks'])} in the manifest for {args.model!r} can be "
                f"served by {pipeline_cls.__name__} with --backend {args.backend}."
            )

        if args.backend == "diffusers":
            backend = DiffusersBackend(args, served)
        else:
            backend = EngineBackend.launch(args, served)

        try:
            app = create_app(
                served,
                GenerationQueue(backend, args.output_dir),
                default_seed=args.default_seed,
                enable_cors=args.enable_cors,
                public_url=args.public_url,
            )
            out.result(
                self.task,
                url=f"http://{args.host}:{args.port}",
                model=served.id,
                backend=args.backend,
                pipeline_class=pipeline_cls.__name__,
                tasks=list(served.tasks),
                capabilities=served.capabilities,
            )
            uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level or "info")
        finally:
            backend.close()
