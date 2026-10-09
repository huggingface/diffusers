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

"""Generative Media Spec (GMS) manifests: loading, derivation from a pipeline signature, and request validation."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from ...pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2VIDEO_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2AUDIO_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_TEXT2VIDEO_PIPELINES_MAPPING,
    AUTO_VIDEO2VIDEO_PIPELINES_MAPPING,
)
from ..run import _AUDIO_INPUT_KEYS, _IMAGE_INPUT_KEYS, _VIDEO_INPUT_KEYS


SPEC_VERSION = "gms/0.1"
MANIFEST_FILENAME = "gms.json"
EXTENSION_KEY = "x-diffusers"

MAX_ADAPTERS = 4
ADAPTER_SOURCES = ("hf",)

# First entry of each tuple is the default when the request names no `output_format`.
OUTPUT_FORMATS = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "video": ("video/mp4",),
    "audio": ("audio/wav",),
}

_ERROR_STATUS = {
    "invalid_request": 400,
    "not_found": 404,
    "rate_limited": 429,
    "internal": 500,
    "not_implemented": 501,
}

_REQUEST_FIELDS = (
    "model",
    "task",
    "preset",
    "inputs",
    "parameters",
    "output_format",
    "response_format",
    "adapters",
)

# GMS name -> pipeline kwargs that carry the same meaning, tried in order before the name itself.
_PARAMETER_KWARGS = {
    "seed": ("generator",),
    "steps": ("num_inference_steps",),
    "num_outputs": ("num_images_per_prompt", "num_videos_per_prompt", "num_waveforms_per_prompt"),
    "fps": ("fps", "frame_rate"),
    "duration": ("num_frames",),
}
_INPUT_KWARGS = {"mask": ("mask_image",)}

# (task, classes registered for it, inputs the task requires, output modality)
_AUTO_TASKS = (
    ("text_to_image", AUTO_TEXT2IMAGE_PIPELINES_MAPPING, (), "image"),
    ("image_to_image", AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, ("image",), "image"),
    ("inpaint", AUTO_INPAINT_PIPELINES_MAPPING, ("image", "mask"), "image"),
    ("text_to_video", AUTO_TEXT2VIDEO_PIPELINES_MAPPING, (), "video"),
    ("image_to_video", AUTO_IMAGE2VIDEO_PIPELINES_MAPPING, ("image",), "video"),
    ("video_to_video", AUTO_VIDEO2VIDEO_PIPELINES_MAPPING, ("video",), "video"),
    ("text_to_audio", AUTO_TEXT2AUDIO_PIPELINES_MAPPING, (), "audio"),
)

_SKIPPED_KWARGS = frozenset({"output_type", "return_dict", "callback_steps", "callback_on_step_end_tensor_inputs"})
_AUDIO_OUTPUT_KWARGS = frozenset({"audio_length_in_s", "audio_end_in_s", "num_waveforms_per_prompt"})
_SCALAR_TYPES = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}
_PYTHON_TYPES = {"number": (int, float), "integer": int, "boolean": bool, "string": str}
_COMPARISONS = {
    "eq": operator.eq,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}
_FRAME_LATTICE = re.compile(r"(\d+)k\+(\d+)")
_HF_ADAPTER_PATH = re.compile(r"hf:[\w.-]+/[\w.-]+")


class GMSError(Exception):
    def __init__(self, code: str, message: str, pointer: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.pointer = pointer
        self.status = _ERROR_STATUS[code]

    def to_dict(self) -> dict[str, Any]:
        error = {"code": self.code, "message": self.message}
        if self.pointer is not None:
            error["pointer"] = self.pointer
        return {"error": error}


@dataclass
class Manifest:
    raw: bytes
    data: dict[str, Any]
    source: str | None = None

    @property
    def hash(self) -> str:
        return f"sha256:{hashlib.sha256(self.raw).hexdigest()}"


@dataclass
class ResolvedRequest:
    task: str
    kwargs: dict[str, Any]
    # pipeline kwarg -> (request input name, media value or list of media values)
    media: dict[str, tuple[str, Any]]
    seed: int | None
    output_formats: dict[str, str]
    response_format: str
    adapters: list[dict[str, Any]] = field(default_factory=list)


def blocks_signature(blocks: Any) -> inspect.Signature:
    """The call signature of a modular pipeline, which takes its keyword arguments from its blocks' inputs."""
    parameters = []
    for input_param in blocks.inputs:
        if input_param.name is None:
            continue
        annotation = inspect.Parameter.empty if input_param.type_hint is None else input_param.type_hint
        parameters.append(
            inspect.Parameter(
                input_param.name, inspect.Parameter.KEYWORD_ONLY, default=input_param.default, annotation=annotation
            )
        )
    return inspect.Signature(parameters)


def load_manifest(
    model: str,
    pipeline_cls: type,
    signature: inspect.Signature,
    kwarg_names: set[str],
    manifest_path: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Manifest:
    if manifest_path is not None:
        return _parse_manifest(Path(manifest_path).read_bytes(), origin=manifest_path)

    if Path(model).is_dir():
        published = Path(model) / MANIFEST_FILENAME
        if published.is_file():
            return _parse_manifest(published.read_bytes(), origin=str(published))
    else:
        try:
            published = hf_hub_download(model, MANIFEST_FILENAME, revision=revision, token=token)
        except EntryNotFoundError:
            published = None
        if published is not None:
            manifest = _parse_manifest(Path(published).read_bytes(), origin=f"{model}/{MANIFEST_FILENAME}")
            manifest.source = f"hf:{model}/{MANIFEST_FILENAME}"
            return manifest

    model_id = Path(model).name if Path(model).is_dir() else model
    data = derive_manifest(model_id, pipeline_cls, signature, kwarg_names)
    return Manifest(raw=json.dumps(data).encode(), data=data)


def _parse_manifest(raw: bytes, origin: str) -> Manifest:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"GMS manifest {origin!r} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        raise SystemExit(f"GMS manifest {origin!r} must be a JSON object with a string `id`.")
    if not isinstance(data.get("tasks"), dict) or not data["tasks"]:
        raise SystemExit(f"GMS manifest {origin!r} declares no `tasks`, so there is nothing to serve.")
    return Manifest(raw=raw, data=data)


def _annotation_schema(annotation: Any, default: Any) -> dict[str, Any] | None:
    """Map a `__call__` annotation to a GMS parameter schema, or `None` when it is not a scalar or list of scalars."""
    if annotation is inspect.Parameter.empty:
        text = type(default).__name__ if default is not None else ""
    elif isinstance(annotation, str):
        text = annotation
    elif type(annotation) is type:
        text = annotation.__name__
    else:
        text = str(annotation)
    text = text.replace("typing.", "")

    def scalar(fragment: str) -> str | None:
        names = set(re.findall(r"[A-Za-z_][\w.]*", fragment))
        names -= {"Optional", "Union", "None", "NoneType", "list", "List"}
        if not names or not names <= set(_SCALAR_TYPES):
            return None
        return next(gms for python, gms in _SCALAR_TYPES.items() if python in names)

    outside_lists = re.sub(r"\b[Ll]ist\[[^\[\]]*\]", "", text)
    if outside_lists == text:
        scalar_type = scalar(text)
        return {"type": scalar_type} if scalar_type else None

    # `str | list[str]` is a scalar that also accepts a batch; only a bare list is an array parameter.
    scalar_type = scalar(text)
    if scalar_type is None:
        return None
    if scalar(outside_lists):
        return {"type": scalar_type}
    return {"type": "array", "items": {"type": scalar_type}}


def derive_manifest(
    model_id: str, pipeline_cls: type, signature: inspect.Signature, kwarg_names: set[str]
) -> dict[str, Any]:
    """Build a server-authored manifest from the pipeline call `signature`, keeping only kwargs in `kwarg_names`."""
    inputs: dict[str, dict[str, Any]] = {}
    parameters: dict[str, dict[str, Any]] = {}
    input_kwargs: dict[str, str] = {}
    parameter_kwargs: dict[str, str] = {}
    media_types = {
        **dict.fromkeys(_IMAGE_INPUT_KEYS, "image"),
        **dict.fromkeys(_VIDEO_INPUT_KEYS, "video"),
        **dict.fromkeys(_AUDIO_INPUT_KEYS, "audio"),
        "mask_image": "mask",
    }
    kwarg_parameters = {kwarg: name for name, kwargs in _PARAMETER_KWARGS.items() for kwarg in kwargs}
    del kwarg_parameters["num_frames"]

    for kwarg, parameter in signature.parameters.items():
        if kwarg == "self" or kwarg in _SKIPPED_KWARGS or kwarg not in kwarg_names:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        has_default = parameter.default is not inspect.Parameter.empty
        default = parameter.default if has_default else None

        if kwarg in media_types:
            name = "mask" if kwarg == "mask_image" else kwarg
            spec: dict[str, Any] = {"type": media_types[kwarg]}
            if not has_default:
                spec["required"] = True
            if re.search(r"\b[Ll]ist\b", str(parameter.annotation)):
                spec["multiple"] = {}
            inputs[name] = spec
            if name != kwarg:
                input_kwargs[name] = kwarg
            continue

        if kwarg == "generator":
            parameters["seed"] = {"type": "integer"}
            parameter_kwargs["seed"] = kwarg
            continue

        schema = _annotation_schema(parameter.annotation, default)
        # Modular blocks often declare `prompt` without a type hint or a default to infer one from.
        if schema is None and kwarg == "prompt" and parameter.annotation is inspect.Parameter.empty:
            schema = {"type": "string"}
        if schema is None:
            continue
        if schema["type"] == "string" and re.fullmatch(r"(.*_)?prompt(_\d+)?", kwarg):
            inputs[kwarg] = {"type": "text", "required": True} if kwarg == "prompt" else {"type": "text"}
            continue

        name = kwarg_parameters.get(kwarg, kwarg)
        if name in parameters:
            continue
        if default is not None and isinstance(default, (bool, int, float, str, list)):
            schema["default"] = default
        parameters[name] = schema
        if name != kwarg:
            parameter_kwargs[name] = kwarg

    registered = [entry for entry in _AUTO_TASKS if pipeline_cls in entry[1].values()]
    names = set(signature.parameters)
    if registered:
        output = registered[0][3]
    elif "num_frames" in names or "Video" in pipeline_cls.__name__:
        output = "video"
    elif names & _AUDIO_OUTPUT_KWARGS:
        output = "audio"
    else:
        output = "image"

    extension: dict[str, Any] = {}
    if input_kwargs:
        extension["inputs"] = input_kwargs
    if parameter_kwargs:
        extension["parameters"] = parameter_kwargs

    def task(required: tuple[str, ...], excluded: set[str]) -> dict[str, Any]:
        task_inputs = {
            name: {**spec, "required": True} if name in required else spec
            for name, spec in inputs.items()
            if name not in excluded
        }
        spec = {"inputs": task_inputs, "outputs": {output: {"type": output}}, "parameters": parameters}
        if extension:
            spec[EXTENSION_KEY] = extension
        return spec

    primary = next((name for name in ("image", "video") if name in inputs), None)
    tasks = {}
    if registered:
        # A class registered for several tasks takes each task's conditioning input optionally; an
        # input another task requires is what tells the tasks apart, so it is left out of the rest.
        for name, _, required, _ in registered:
            if not set(required) <= set(inputs):
                continue
            others = {key for other in registered if other[0] != name for key in other[2]}
            tasks[name] = task(required, others - set(required))
    elif primary is None:
        tasks[f"text_to_{output}"] = task((), set())
    elif inputs[primary].get("required"):
        tasks[f"{primary}_to_{output}"] = task((), set())
    else:
        tasks[f"text_to_{output}"] = task((), {primary})
        tasks[f"{primary}_to_{output}"] = task((primary,), set())

    modalities_in = sorted({"image" if s["type"] == "mask" else s["type"] for s in inputs.values()} - {"text"})
    manifest: dict[str, Any] = {
        "id": model_id,
        "modalities": {"in": modalities_in or ["text"], "out": [output]},
        "tasks": tasks,
    }
    if hasattr(pipeline_cls, "load_lora_weights"):
        manifest["adapters"] = {"lora": {"targets": list(getattr(pipeline_cls, "_lora_loadable_modules", []))}}
    return manifest


def _bind(name: str, aliases: dict[str, tuple[str, ...]], declared: dict[str, str], kwarg_names: set[str]):
    candidates = (declared[name],) if name in declared else (*aliases.get(name, ()), name)
    return next((kwarg for kwarg in candidates if kwarg in kwarg_names), None)


def _frame_lattice(spec: dict[str, Any]) -> tuple[int, int, float] | None:
    lattice = spec.get("lattice") or {}
    match = _FRAME_LATTICE.fullmatch(str(lattice.get("frames", "")).replace(" ", ""))
    if match is None or not isinstance(lattice.get("fps"), (int, float)):
        return None
    return int(match.group(1)), int(match.group(2)), lattice["fps"]


def _is_active(condition: dict[str, Any], values: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(_is_active(item, values) for item in condition["all"])
    actual = values.get(condition["parameter"])
    for name, compare in _COMPARISONS.items():
        if name not in condition:
            continue
        if actual is None or isinstance(actual, str) != isinstance(condition[name], str):
            return False
        if not compare(actual, condition[name]):
            return False
    return True


def _check_value(label: str, value: Any, spec: dict[str, Any], pointer: str) -> None:
    kind = spec.get("type")
    if kind == "array":
        if not isinstance(value, list):
            raise GMSError("invalid_request", f"{label} must be an array, got {value!r}", pointer)
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            raise GMSError("invalid_request", f"{label} accepts at most {spec['maxItems']} items", pointer)
        for index, item in enumerate(value):
            _check_value(f"{label}[{index}]", item, spec["items"], f"{pointer}/{index}")
        return
    if kind == "object":
        if not isinstance(value, dict):
            raise GMSError("invalid_request", f"{label} must be an object, got {value!r}", pointer)
        for key, item in value.items():
            if key not in spec["properties"]:
                raise GMSError("invalid_request", f"{label} has no property {key!r}", f"{pointer}/{key}")
            _check_value(f"{label}.{key}", item, spec["properties"][key], f"{pointer}/{key}")
        return

    expected = _PYTHON_TYPES.get(kind)
    is_bool = isinstance(value, bool)
    if expected is not None and (not isinstance(value, expected) or (is_bool and kind != "boolean")):
        raise GMSError("invalid_request", f"{label} must be of type {kind}, got {value!r}", pointer)
    if isinstance(value, float) and not math.isfinite(value):
        raise GMSError("invalid_request", f"{label} must be a finite number, got {value}", pointer)
    if "enum" in spec and value not in spec["enum"]:
        raise GMSError("invalid_request", f"{label} must be one of {spec['enum']}, got {value!r}", pointer)
    if "min" in spec and value < spec["min"]:
        raise GMSError("invalid_request", f"{label} must be at least {spec['min']}, got {value}", pointer)
    if "max" in spec and value > spec["max"]:
        raise GMSError("invalid_request", f"{label} must be at most {spec['max']}, got {value}", pointer)
    if "multiple_of" in spec:
        quotient = value / spec["multiple_of"]
        if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
            raise GMSError(
                "invalid_request", f"{label} must be a multiple of {spec['multiple_of']}, got {value}", pointer
            )


def _check_media(name: str, value: Any, spec: dict[str, Any], pointer: str) -> None:
    if not isinstance(value, dict):
        raise GMSError("invalid_request", f"{name} must be a media value object, got {value!r}", pointer)
    if value.get("type") != spec["type"]:
        raise GMSError(
            "invalid_request", f"{name} must have type {spec['type']!r}, got {value.get('type')!r}", f"{pointer}/type"
        )
    keys = set(value)
    if keys == {"type", "base64", "media_type"}:
        if not isinstance(value["base64"], str) or not isinstance(value["media_type"], str):
            raise GMSError("invalid_request", f"{name} base64 and media_type must be strings", pointer)
        return
    if "url" not in keys or not keys <= {"type", "url", "media_type"}:
        raise GMSError(
            "invalid_request",
            f"{name} must be either {{type, url, media_type?}} or {{type, base64, media_type}}, got keys {sorted(keys)}",
            pointer,
        )
    # A URL media value is fetched by the server, so anything but http(s) would read the server's own filesystem.
    if not isinstance(value["url"], str) or not value["url"].startswith(("http://", "https://")):
        raise GMSError("invalid_request", f"{name} url must be an http(s) URL", f"{pointer}/url")


class ServedModel:
    """One manifest bound to the kwargs a backend accepts: the discovery document and request resolution."""

    def __init__(
        self,
        manifest: Manifest,
        kwarg_names: set[str],
        output_types: set[str],
        capabilities: list[str],
        revision: str | None = None,
    ):
        self.manifest = manifest
        self.id = manifest.data["id"]
        self.capabilities = capabilities
        self.revision = revision
        self.adapters = manifest.data.get("adapters", {})
        self._kwarg_names = kwarg_names

        self.tasks: dict[str, dict[str, Any]] = {}
        self._inputs: dict[str, dict[str, str | None]] = {}
        self._parameters: dict[str, dict[str, str | None]] = {}
        self._presets: dict[str, list[str]] = {}
        for name, task in manifest.data["tasks"].items():
            declared = task.get(EXTENSION_KEY, {})
            inputs = {
                key: _bind(key, _INPUT_KWARGS, declared.get("inputs", {}), kwarg_names) for key in task["inputs"]
            }
            parameters = {
                key: _bind(key, _PARAMETER_KWARGS, declared.get("parameters", {}), kwarg_names)
                for key in task.get("parameters", {})
            }
            if "duration" in parameters and _frame_lattice(task["parameters"]["duration"]) is None:
                parameters["duration"] = None
            if any(inputs[key] is None for key, spec in task["inputs"].items() if spec.get("required")):
                continue
            if not any(spec["type"] in output_types for spec in task["outputs"].values()):
                continue
            self.tasks[name] = task
            self._inputs[name] = inputs
            self._parameters[name] = parameters
            self._presets[name] = [
                preset
                for preset, body in task.get("presets", {}).items()
                if all(parameters.get(key) is not None for key in body.get("parameters", {}))
            ]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modalities": self.manifest.data.get("modalities", {}),
            "tasks": list(self.tasks),
        }

    def overlay(self) -> dict[str, Any]:
        tasks = {}
        limits = {}
        for name, task in self.tasks.items():
            tasks[name] = {"presets": self._presets[name]} if "presets" in task else {}
            disabled = {key: False for key, kwarg in self._inputs[name].items() if kwarg is None}
            if disabled:
                limits[name] = {"inputs": disabled}
        overlay: dict[str, Any] = {"tasks": tasks, "capabilities": self.capabilities}
        if limits:
            overlay["limits"] = limits
        if "adapters" in self.capabilities:
            overlay["adapters"] = {"max": MAX_ADAPTERS, "sources": list(ADAPTER_SOURCES)}
        return overlay

    def discovery_document(self) -> bytes:
        # The manifest bytes are spliced in untouched: the spec defines verification as a hash of the served bytes.
        head: dict[str, Any] = {"spec": SPEC_VERSION}
        if self.manifest.source is not None:
            head["source"] = self.manifest.source
            head["hash"] = self.manifest.hash
        prefix = json.dumps(head)[:-1].encode()
        return (
            prefix
            + b', "manifest": '
            + self.manifest.raw
            + b', "server": '
            + json.dumps(self.overlay()).encode()
            + b"}"
        )

    def resolve(self, body: Any) -> ResolvedRequest:
        if not isinstance(body, dict):
            raise GMSError("invalid_request", "request body must be a JSON object")
        for key in body:
            if key not in _REQUEST_FIELDS:
                raise GMSError("invalid_request", f"unknown request field {key!r}", f"/{key}")
        if not isinstance(body.get("model"), str):
            raise GMSError("invalid_request", "model is required and must be a string", "/model")
        if body["model"] != self.id:
            raise GMSError("not_found", f"model {body['model']!r} is not served here; this server hosts {self.id!r}")
        inputs = body.get("inputs")
        if not isinstance(inputs, dict):
            raise GMSError("invalid_request", "inputs is required and must be an object", "/inputs")

        name = self._select_task(body.get("task"), inputs)
        task = self.tasks[name]
        values, explicit = self._resolve_parameters(name, task, body)

        kwargs: dict[str, Any] = {}
        media: dict[str, tuple[str, Any]] = {}
        for key, value in inputs.items():
            pointer = f"/inputs/{key}"
            spec = task["inputs"].get(key)
            if spec is None:
                raise GMSError("invalid_request", f"task {name!r} declares no input {key!r}", pointer)
            kwarg = self._inputs[name][key]
            if kwarg is None:
                raise GMSError("invalid_request", f"input {key!r} is disabled on this server", pointer)
            if "active_when" in spec and not _is_active(spec["active_when"], values):
                raise GMSError(
                    "invalid_request", f"input {key!r} is inactive unless {json.dumps(spec['active_when'])}", pointer
                )
            if spec["type"] == "text":
                if not isinstance(value, str):
                    raise GMSError("invalid_request", f"{key} must be a string, got {value!r}", pointer)
                kwargs[kwarg] = value
                continue
            if isinstance(value, list):
                if "multiple" not in spec:
                    raise GMSError("invalid_request", f"{key} accepts a single media value, not a list", pointer)
                limit = spec["multiple"].get("max")
                if limit is not None and len(value) > limit:
                    raise GMSError(
                        "invalid_request", f"{key} accepts at most {limit} items, got {len(value)}", pointer
                    )
                for index, item in enumerate(value):
                    _check_media(key, item, spec, f"{pointer}/{index}")
            else:
                _check_media(key, value, spec, pointer)
            media[kwarg] = (key, value)
        for key, spec in task["inputs"].items():
            if spec.get("required") and key not in inputs:
                raise GMSError("invalid_request", f"task {name!r} requires input {key!r}", f"/inputs/{key}")

        seed = None
        for key, value in values.items():
            spec = task["parameters"][key]
            kwarg = self._parameters[name][key]
            pointer = f"/parameters/{key}"
            if "active_when" in spec and not _is_active(spec["active_when"], values):
                if key in explicit:
                    raise GMSError(
                        "invalid_request",
                        f"parameter {key!r} is inactive unless {json.dumps(spec['active_when'])}",
                        pointer,
                    )
                continue
            if kwarg is None:
                # Clients that render every control resend defaults, so a default on a parameter this
                # backend cannot set is the one supplied value that changes nothing by being skipped.
                if key in explicit and value != spec.get("default"):
                    raise GMSError(
                        "not_implemented", f"parameter {key!r} is not supported by this server's backend", pointer
                    )
                continue
            if key == "seed":
                if not 0 <= value < 2**63:
                    raise GMSError("invalid_request", f"seed must be between 0 and 2**63 - 1, got {value}", pointer)
                seed = value
                continue
            if key == "duration":
                lattice_fps = spec["lattice"]["fps"]
                if values.get("fps", lattice_fps) != lattice_fps:
                    raise GMSError(
                        "invalid_request",
                        f"duration is declared on a {lattice_fps} fps frame lattice; fps must be {lattice_fps} "
                        f"when duration is set, got {values['fps']}",
                        "/parameters/fps",
                    )
                kwargs[kwarg] = self._duration_to_frames(value, spec, pointer)
                fps_kwarg = _bind("fps", _PARAMETER_KWARGS, {}, self._kwarg_names)
                if fps_kwarg is not None:
                    kwargs[fps_kwarg] = lattice_fps
                continue
            kwargs[kwarg] = value

        output_formats = {kind: formats[0] for kind, formats in OUTPUT_FORMATS.items()}
        output_format = body.get("output_format")
        if output_format is not None:
            kind = output_format.split("/")[0] if isinstance(output_format, str) else None
            produced = {spec["type"] for spec in task["outputs"].values()}
            if kind not in produced or output_format not in OUTPUT_FORMATS.get(kind, ()):
                supported = sorted(f for k in produced for f in OUTPUT_FORMATS.get(k, ()))
                raise GMSError(
                    "invalid_request",
                    f"output_format must be one of {supported} for task {name!r}, got {output_format!r}",
                    "/output_format",
                )
            output_formats[kind] = output_format

        response_format = body.get("response_format", "url")
        if response_format not in ("url", "b64_json"):
            raise GMSError(
                "invalid_request",
                f"response_format must be 'url' or 'b64_json', got {response_format!r}",
                "/response_format",
            )

        return ResolvedRequest(
            task=name,
            kwargs=kwargs,
            media=media,
            seed=seed,
            output_formats=output_formats,
            response_format=response_format,
            adapters=self._resolve_adapters(body.get("adapters")),
        )

    def seeded(self, task: str) -> bool:
        return self._parameters[task].get("seed") is not None

    def _select_task(self, requested: Any, inputs: dict[str, Any]) -> str:
        if requested is not None:
            if not isinstance(requested, str) or requested not in self.tasks:
                raise GMSError(
                    "invalid_request",
                    f"task {requested!r} is not hosted for {self.id!r}; hosted tasks: {list(self.tasks)}",
                    "/task",
                )
            return requested

        supplied = set(inputs)
        candidates = []
        for name, task in self.tasks.items():
            accepted = {key for key, kwarg in self._inputs[name].items() if kwarg is not None}
            required = {key for key, spec in task["inputs"].items() if spec.get("required")}
            if required <= supplied <= accepted:
                candidates.append(name)
        if len(candidates) == 1:
            return candidates[0]
        if len(self.tasks) == 1:
            return next(iter(self.tasks))
        raise GMSError(
            "invalid_request",
            f"could not infer the task from inputs {sorted(supplied)}; set `task` to one of {list(self.tasks)}",
            "/task",
        )

    def _resolve_parameters(self, name: str, task: dict[str, Any], body: dict[str, Any]):
        declared = task.get("parameters", {})
        explicit = body.get("parameters", {})
        if not isinstance(explicit, dict):
            raise GMSError("invalid_request", "parameters must be an object", "/parameters")
        for key, value in explicit.items():
            if key not in declared:
                raise GMSError("invalid_request", f"task {name!r} declares no parameter {key!r}", f"/parameters/{key}")
            _check_value(key, value, declared[key], f"/parameters/{key}")

        values = {key: spec["default"] for key, spec in declared.items() if "default" in spec}
        preset = body.get("preset")
        if preset is not None:
            if preset not in self._presets[name]:
                raise GMSError(
                    "invalid_request",
                    f"preset {preset!r} is not supported for task {name!r}; supported presets: {self._presets[name]}",
                    "/preset",
                )
            values.update(task["presets"][preset].get("parameters", {}))
        values.update(explicit)
        return values, set(explicit)

    def _duration_to_frames(self, seconds: float, spec: dict[str, Any], pointer: str) -> int:
        step, offset, fps = _frame_lattice(spec)
        position = (seconds * fps - offset) / step
        snap = spec["lattice"].get("snap", "nearest")
        if snap == "reject" and not math.isclose(position, round(position), abs_tol=1e-9):
            raise GMSError(
                "invalid_request",
                f"duration {seconds} is off the {spec['lattice']['frames']} frame lattice at {fps} fps",
                pointer,
            )
        index = math.ceil(position - 1e-9) if snap == "up" else round(position)
        return step * max(index, 0) + offset

    def _resolve_adapters(self, adapters: Any) -> list[dict[str, Any]]:
        if adapters is None:
            return []
        if "adapters" not in self.capabilities:
            raise GMSError("not_implemented", "this server does not apply adapters", "/adapters")
        if not isinstance(adapters, list):
            raise GMSError("invalid_request", "adapters must be an array", "/adapters")
        if len(adapters) > MAX_ADAPTERS:
            raise GMSError(
                "invalid_request", f"at most {MAX_ADAPTERS} adapters per request, got {len(adapters)}", "/adapters"
            )
        for index, adapter in enumerate(adapters):
            pointer = f"/adapters/{index}"
            if not isinstance(adapter, dict):
                raise GMSError("invalid_request", "an adapter must be an object", pointer)
            for key in adapter:
                if key not in ("type", "path", "scale", "targets"):
                    raise GMSError("invalid_request", f"unknown adapter field {key!r}", f"{pointer}/{key}")
            if not isinstance(adapter.get("type"), str) or adapter["type"] not in self.adapters:
                raise GMSError(
                    "invalid_request",
                    f"adapter type must be one of {list(self.adapters)}, got {adapter.get('type')!r}",
                    f"{pointer}/type",
                )
            path = adapter.get("path")
            # Anything but a repo id would reach `load_lora_weights` as a path on the server's own disk.
            if not isinstance(path, str) or not _HF_ADAPTER_PATH.fullmatch(path):
                raise GMSError(
                    "invalid_request",
                    f"adapter path must be a Hub repo id of the form 'hf:<namespace>/<name>', got {path!r}",
                    f"{pointer}/path",
                )
            scale = adapter.get("scale", 1.0)
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise GMSError("invalid_request", f"adapter scale must be a number, got {scale!r}", f"{pointer}/scale")
            valid_targets = self.adapters[adapter["type"]].get("targets", [])
            targets = adapter.get("targets", {})
            if not isinstance(targets, dict):
                raise GMSError("invalid_request", "adapter targets must be an object", f"{pointer}/targets")
            for target, target_scale in targets.items():
                if target not in valid_targets:
                    raise GMSError(
                        "invalid_request",
                        f"adapter target must be one of {valid_targets}, got {target!r}",
                        f"{pointer}/targets/{target}",
                    )
                if isinstance(target_scale, bool) or not isinstance(target_scale, (int, float)):
                    raise GMSError(
                        "invalid_request",
                        f"adapter target scale must be a number, got {target_scale!r}",
                        f"{pointer}/targets/{target}",
                    )
        return adapters
