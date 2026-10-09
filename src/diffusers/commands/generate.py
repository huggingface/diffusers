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

"""`diffusers-cli generate` — send a generation to a Generative Media Spec (GMS) server and save its outputs.

The client side of `diffusers-cli serve`: it loads no pipeline. It reads the model's discovery document, submits the
request, polls the generation until it finishes, and downloads the files.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter, _SubParsersAction
from pathlib import Path
from typing import Any

from huggingface_hub.cli._output import out
from huggingface_hub.utils import httpx

from . import BaseDiffusersCLICommand


DEFAULT_OUTPUT_DIR = str(Path.home() / ".diffusers" / "cli" / "generate" / "outputs")
_MEDIA_TYPES = ("image", "mask", "video", "audio")
_POLL_SECONDS = 1.0
_TIMEOUT_SECONDS = 60.0


def _parse_json(raw: str | None, flag: str, expected: type) -> Any:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{flag} must be valid JSON: {e}") from e
    if not isinstance(parsed, expected):
        raise SystemExit(f"{flag} must decode to a JSON {'object' if expected is dict else 'array'}.")
    return parsed


def _media_value(name: str, media_type: str, value: Any) -> Any:
    """Turn a URL or local path into a GMS media value; anything else is sent as given."""
    if isinstance(value, list):
        return [_media_value(name, media_type, item) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith(("http://", "https://")):
        return {"type": media_type, "url": value}

    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"Input {name!r} is neither an http(s) URL nor an existing file: {value!r}")
    media = {"type": media_type, "base64": base64.b64encode(path.read_bytes()).decode()}
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed is not None:
        media["media_type"] = guessed
    return media


class GenerateCommand(BaseDiffusersCLICommand):
    task = "generate"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        epilog = (
            "Examples\n"
            '  $ diffusers-cli generate --url http://localhost:8000 --inputs \'{"prompt": "a cat on the moon"}\'\n'
            "  $ diffusers-cli generate --url http://localhost:8000 \\\n"
            '      --inputs \'{"prompt": "make the fur grey", "image": "cat.png"}\' \\\n'
            '      --parameters \'{"steps": 8, "seed": 0}\' -o outputs/\n'
            '  $ diffusers-cli generate --sandbox-id <id> --inputs \'{"prompt": "a cat on the moon"}\'\n'
            "\n"
            "Learn more\n"
            "  Use `diffusers-cli <command> --help` for more information about a command.\n"
            "  Read the documentation at https://huggingface.co/docs/diffusers\n"
        )

        parser: ArgumentParser = subparsers.add_parser(
            "generate",
            help="Send a generation to a Generative Media Spec server and save its outputs.",
            usage="\n  diffusers-cli generate (--url <server> | --sandbox-id <id>) --inputs <json> [options]",
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter,
        )
        parser._optionals.title = "Options"
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument("--url", default=None, help="Base URL of the server, e.g. http://localhost:8000.")
        target.add_argument(
            "--sandbox-id",
            default=None,
            help="Id of a sandbox started with `diffusers-cli serve --remote`, reached through the sandbox proxy.",
        )
        parser.add_argument(
            "--model", "-m", default=None, help="Model id to request. Defaults to the server's only model."
        )
        parser.add_argument(
            "--inputs",
            required=True,
            metavar="JSON",
            help=(
                "JSON object of the task's inputs. An image, mask, video or audio input given as an http(s) URL "
                "or a local file path is converted to a media value; local files are sent inline as base64."
            ),
        )
        parser.add_argument("--parameters", default=None, metavar="JSON", help="JSON object of task parameters.")
        parser.add_argument(
            "--task", default=None, help="Task to run. Defaults to the one the server infers from the inputs."
        )
        parser.add_argument("--preset", default=None, help="Named preset of the task to start from.")
        parser.add_argument(
            "--adapters",
            default=None,
            metavar="JSON",
            help='JSON array of adapters to apply, e.g. \'[{"type": "lora", "path": "hf:org/name", "scale": 0.8}]\'.',
        )
        parser.add_argument("--output-format", default=None, help="Media type to encode outputs as, e.g. image/jpeg.")
        parser.add_argument(
            "--output",
            "-o",
            default=None,
            help=f"Directory to save outputs in. Defaults to {DEFAULT_OUTPUT_DIR}/<generation id>/.",
        )
        parser.add_argument(
            "--token",
            default=None,
            help=(
                "With --url: bearer token sent in the Authorization header; none is sent by default. "
                "With --sandbox-id: Hugging Face token used to reach the sandbox; defaults to the logged-in one."
            ),
        )
        parser.set_defaults(func=GenerateCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        args = self.args
        inputs = _parse_json(args.inputs, "--inputs", dict)
        parameters = _parse_json(args.parameters, "--parameters", dict)
        adapters = _parse_json(args.adapters, "--adapters", list)

        with self._client() as client:
            model = args.model or self._only_model(client)
            document = self._request(client, "GET", f"/v1/models/{model}")
            tasks = document["manifest"]["tasks"]
            if args.task is not None and args.task not in tasks:
                raise SystemExit(f"Model {model!r} declares no task {args.task!r}; its tasks are {list(tasks)}.")

            declared = [tasks[args.task]] if args.task is not None else list(tasks.values())
            input_types = {name: spec.get("type") for task in declared for name, spec in task["inputs"].items()}
            body: dict[str, Any] = {"model": model, "inputs": {}}
            for name, value in inputs.items():
                if input_types.get(name) in _MEDIA_TYPES:
                    value = _media_value(name, input_types[name], value)
                body["inputs"][name] = value
            optional = {
                "task": args.task,
                "preset": args.preset,
                "parameters": parameters,
                "adapters": adapters,
                "output_format": args.output_format,
            }
            body.update({key: value for key, value in optional.items() if value is not None})

            generation = self._request(client, "POST", "/v1/generations", json=body)
            try:
                while generation["status"] in ("queued", "generating"):
                    time.sleep(_POLL_SECONDS)
                    generation = self._request(client, "GET", f"/v1/generations/{generation['id']}")
            except KeyboardInterrupt:
                # Without this the server keeps generating for a client that has gone away.
                self._request(client, "POST", f"/v1/generations/{generation['id']}/cancel")
                raise
            if generation["status"] != "complete":
                raise SystemExit(f"Generation {generation['id']} ended with status {generation['status']!r}.")

            directory = Path(args.output or Path(DEFAULT_OUTPUT_DIR) / generation["id"]).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            saved = []
            for name, artifacts in generation["outputs"].items():
                for index, artifact in enumerate(artifacts):
                    path = directory / f"{name}-{index}.{artifact['media_type'].split('/')[1]}"
                    path.write_bytes(self._artifact_bytes(client, artifact))
                    saved.append(str(path))

        out.result(
            self.task,
            url=str(client.base_url),
            model=model,
            generation=generation["id"],
            seed=generation.get("seed"),
            outputs=saved,
        )

    def _client(self) -> httpx.Client:
        if self.args.sandbox_id is not None:
            from huggingface_hub import Sandbox

            from .serve.remote import SANDBOX_PORT

            sbx = Sandbox.connect(self.args.sandbox_id, token=self.args.token)
            return httpx.Client(
                base_url=sbx.proxy_url_for(SANDBOX_PORT),
                headers=sbx.proxy_headers,
                timeout=_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
        headers = {"Authorization": f"Bearer {self.args.token}"} if self.args.token else {}
        return httpx.Client(base_url=self.args.url, headers=headers, timeout=_TIMEOUT_SECONDS, follow_redirects=True)

    def _request(self, client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise SystemExit(f"Could not reach {client.base_url}: {type(e).__name__}: {e}") from e
        if response.status_code < 400:
            return response.json()

        try:
            error = response.json()["error"]
            message = f"{error['code']}: {error['message']}"
            if error.get("pointer"):
                message += f" (at {error['pointer']})"
        except (ValueError, KeyError, TypeError):
            message = response.text
        raise SystemExit(f"{method} {path} failed with HTTP {response.status_code}: {message}")

    def _only_model(self, client: httpx.Client) -> str:
        models = [model["id"] for model in self._request(client, "GET", "/v1/models")["data"]]
        if len(models) != 1:
            raise SystemExit(f"{client.base_url} serves {len(models)} models ({models}); pick one with --model.")
        return models[0]

    def _artifact_bytes(self, client: httpx.Client, artifact: dict[str, Any]) -> bytes:
        if "base64" in artifact:
            return base64.b64decode(artifact["base64"])
        try:
            # The token is for the server; an output hosted elsewhere is fetched without it.
            if httpx.URL(artifact["url"]).host == client.base_url.host:
                response = client.get(artifact["url"])
            else:
                response = httpx.get(artifact["url"], timeout=_TIMEOUT_SECONDS, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SystemExit(f"Could not download output {artifact['url']}: {type(e).__name__}: {e}") from e
        return response.content
