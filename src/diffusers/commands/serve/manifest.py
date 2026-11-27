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

"""Generative Media Spec (GMS) manifests.

A manifest describes what this server offers: its tasks with their inputs and parameters, and its capabilities. It is
loaded from a published `gms.json` or derived from the pipeline signature, checked against the pipeline at startup,
served for discovery, and used to validate each request and map it to the pipeline call.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import os
import random
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import HFValidationError, validate_repo_id

from ...pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_IMAGE2VIDEO_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2AUDIO_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
    AUTO_TEXT2VIDEO_PIPELINES_MAPPING,
    AUTO_VIDEO2VIDEO_PIPELINES_MAPPING,
)
from ...utils import logging
from ..run import _AUDIO_INPUT_KEYS, _IMAGE_INPUT_KEYS, _VIDEO_INPUT_KEYS


logger = logging.get_logger("diffusers-cli/serve")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEC_VERSION = "gms/0.1"
MANIFEST_FILENAME = "gms.json"
EXTENSION_KEY = "x-diffusers"

# The frame rate videos are encoded at when the pipeline gives none. Native rates range from 8 fps (CogVideoX) to 30
# (Mochi), so this is a guess.
DEFAULT_FPS = 16

# The fields of a pipeline's output that hold each output type, tried in order.
OUTPUT_FIELDS = {
    "image": ("images", "image"),
    "video": ("frames", "videos", "video"),
    "audio": ("audios", "audio", "sound"),
}
# First entry of each tuple is the default when the request names no `output_format`.
OUTPUT_FORMATS = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "video": ("video/mp4",),
    "audio": ("audio/wav",),
}

_ERROR_STATUS = {
    "invalid_request": 400,
    "not_found": 404,
    "internal": 500,
    "not_implemented": 501,
}
_REQUEST_FIELDS = ("model", "task", "preset", "inputs", "parameters", "output_format", "response_format", "adapters")
_ADAPTER_FIELDS = ("type", "path", "scale", "targets")

# GMS name -> pipeline kwargs that carry the same meaning, tried in order before the name itself.
_PARAMETER_KWARGS = {
    "seed": ("generator",),
    "steps": ("num_inference_steps",),
    "num_outputs": ("num_images_per_prompt", "num_videos_per_prompt", "num_waveforms_per_prompt"),
    "fps": ("frame_rate",),
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

_MEDIA_KWARGS = {
    **dict.fromkeys(_IMAGE_INPUT_KEYS, "image"),
    **dict.fromkeys(_VIDEO_INPUT_KEYS, "video"),
    **dict.fromkeys(_AUDIO_INPUT_KEYS, "audio"),
    "mask_image": "mask",
}
# Pipeline kwargs that take text, served as `text` inputs. `prompt` is the one a task requires.
_TEXT_KWARGS = frozenset(
    {
        "prompt",
        "prompt_2",
        "prompt_3",
        "prompt_4",
        "negative_prompt",
        "negative_prompt_2",
        "negative_prompt_3",
        "negative_prompt_4",
        "neg_prompt",
        "source_prompt",
        "editing_prompt",
        "content_prompt",
        "system_prompt",
        "pe_system_prompt",
        "task_prompt",
    }
)
_SKIPPED_KWARGS = frozenset({"output_type", "return_dict", "callback_steps", "callback_on_step_end_tensor_inputs"})
_AUDIO_OUTPUT_KWARGS = frozenset({"audio_length_in_s", "audio_end_in_s", "num_waveforms_per_prompt"})
# In order of preference when an annotation allows several, so `int | float` is a number.
_SCALAR_TYPES = {float: "number", int: "integer", bool: "boolean", str: "string"}
_PYTHON_TYPES = {"number": (int, float), "integer": int, "boolean": bool, "string": str}
_COMPARISONS = {
    "eq": operator.eq,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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
class ResolvedRequest:
    task: str
    kwargs: dict[str, Any]
    # pipeline kwarg -> (request input name, media type, media value or list of media values)
    media: dict[str, tuple[str, str, Any]]
    seed: int | None
    # output name -> (output type, media type to encode it as)
    outputs: dict[str, tuple[str, str]]
    # The frame rate video outputs are encoded at, or None when the task produces no video.
    fps: float | None
    response_format: str
    adapters: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Loading a manifest
# ---------------------------------------------------------------------------


def load_manifest(
    model: str, manifest_path: str | None = None, revision: str | None = None, token: str | None = None
) -> tuple[bytes, dict[str, Any], str | None] | None:
    """Read the manifest passed with `--manifest`, else the repo's `gms.json`, as `(raw bytes, data, source)`.

    Returns None when there is neither, so the caller derives one from the pipeline instead.
    """
    source = None
    if manifest_path is not None:
        raw, origin = Path(manifest_path).read_bytes(), manifest_path
    elif Path(model).is_dir():
        published = Path(model) / MANIFEST_FILENAME
        if not published.is_file():
            return None
        raw, origin = published.read_bytes(), str(published)
    else:
        try:
            published = hf_hub_download(model, MANIFEST_FILENAME, revision=revision, token=token)
        except EntryNotFoundError:
            return None
        raw, origin = Path(published).read_bytes(), f"{model}/{MANIFEST_FILENAME}"
        source = f"hf:{model}/{MANIFEST_FILENAME}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"GMS manifest {origin!r} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        raise SystemExit(f"GMS manifest {origin!r} must be a JSON object with a string `id`.")
    if not isinstance(data.get("tasks"), dict) or not data["tasks"]:
        raise SystemExit(f"GMS manifest {origin!r} declares no `tasks`, so there is nothing to serve.")
    return raw, data, source


# ---------------------------------------------------------------------------
# Deriving a manifest from the pipeline signature
# ---------------------------------------------------------------------------


def blocks_signature(blocks: Any) -> inspect.Signature:
    """The call signature of a modular pipeline, which takes its keyword arguments from its blocks' inputs."""
    parameters = []
    for input_param in blocks.inputs:
        if input_param.name is None:
            continue
        annotation = inspect.Parameter.empty if input_param.type_hint is None else input_param.type_hint
        # A required input is given no default, as a required argument of a `__call__` has none.
        default = inspect.Parameter.empty if input_param.required else input_param.default
        parameters.append(
            inspect.Parameter(input_param.name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )
    return inspect.Signature(parameters)


def derive_manifest(
    model_id: str,
    pipeline_cls: type,
    signature: inspect.Signature,
    capabilities: list[str],
    output_fields: set[str] | None = None,
    workflow_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a server-authored manifest from the pipeline call `signature`, declaring the server's `capabilities`.

    `output_fields`, the fields of the pipeline's output when it annotates them, decides the output type. A modular
    pipeline's `workflow_map` gives its tasks.
    """
    # Sort the call kwargs into GMS inputs and parameters, and record the ones whose GMS name differs.
    inputs: dict[str, dict[str, Any]] = {}
    parameters: dict[str, dict[str, Any]] = {}
    input_kwargs: dict[str, str] = {}
    parameter_kwargs: dict[str, str] = {}
    # Pipeline kwarg -> its GMS parameter name.
    kwarg_parameters = {kwarg: name for name, kwargs in _PARAMETER_KWARGS.items() for kwarg in kwargs}

    for kwarg, parameter in signature.parameters.items():
        if kwarg == "self" or kwarg in _SKIPPED_KWARGS:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        has_default = parameter.default is not inspect.Parameter.empty
        default = parameter.default if has_default else None

        if kwarg in _MEDIA_KWARGS:
            name = "mask" if kwarg == "mask_image" else kwarg
            spec: dict[str, Any] = {"type": _MEDIA_KWARGS[kwarg]}
            if not has_default:
                spec["required"] = True
            if re.search(r"\b[Ll]ist\b", str(parameter.annotation)):
                spec["multiple"] = {}
            inputs[name] = spec
            if name != kwarg:
                input_kwargs[name] = kwarg
            continue

        if kwarg in _TEXT_KWARGS:
            inputs[kwarg] = {"type": "text", "required": True} if kwarg == "prompt" else {"type": "text"}
            continue

        if kwarg == "generator":
            parameters["seed"] = {"type": "integer"}
            parameter_kwargs["seed"] = kwarg
            continue

        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty and default is not None:
            annotation = type(default)
        schema = _parameter_schema(annotation)
        if schema is None:
            logger.debug(f"Leaving {kwarg!r} out of the manifest: its type {annotation} has no GMS parameter type.")
            continue

        name = kwarg_parameters.get(kwarg, kwarg)
        if name in parameters:
            logger.debug(f"Leaving {kwarg!r} out of the manifest: parameter {name!r} already sets another argument.")
            continue
        if isinstance(default, (bool, int, float, str, list)):
            schema["default"] = default
        parameters[name] = schema
        if name != kwarg:
            parameter_kwargs[name] = kwarg

    extension: dict[str, Any] = {}
    if input_kwargs:
        extension["inputs"] = input_kwargs
    if parameter_kwargs:
        extension["parameters"] = parameter_kwargs

    auto_tasks = [entry for entry in _AUTO_TASKS if pipeline_cls in entry[1].values()]
    returned = [kind for kind, fields in OUTPUT_FIELDS.items() if output_fields and set(fields) & output_fields]
    if returned:
        output = returned[0]
    elif auto_tasks:
        output = auto_tasks[0][3]
    elif "num_frames" in signature.parameters or "Video" in pipeline_cls.__name__:
        output = "video"
    elif set(signature.parameters) & _AUDIO_OUTPUT_KWARGS:
        output = "audio"
    else:
        output = "image"
    outputs = returned or [output]
    # (task, inputs it requires, inputs that tell it apart from the other tasks)
    registered = [(name, required, set(required)) for name, _, required, _ in auto_tasks]
    if workflow_map:
        registered = _workflow_tasks(workflow_map, inputs, output)
        if not registered:
            raise SystemExit(
                f"No workflow of {pipeline_cls.__name__} in {list(workflow_map)} can be served: each is triggered by "
                "something other than a text, image, video, audio or mask input. Publish a `gms.json` with inputs the "
                "server can take, as in the Custom inputs section of the `diffusers serve` docs."
            )

    def task(required: tuple[str, ...], excluded: set[str]) -> dict[str, Any]:
        task_inputs = {
            name: {**spec, "required": True} if name in required else spec
            for name, spec in inputs.items()
            if name not in excluded
        }
        spec = {"inputs": task_inputs, "outputs": {kind: {"type": kind} for kind in outputs}, "parameters": parameters}
        if extension:
            spec[EXTENSION_KEY] = extension
        return spec

    primary = None
    if "image" in inputs:
        primary = "image"
    elif "video" in inputs:
        primary = "video"
    tasks = {}
    if registered:
        # A class registered for several tasks takes each task's conditioning input optionally; an
        # input another task requires is what tells the tasks apart, so it is left out of the rest.
        for name, required, _ in registered:
            if not set(required) <= set(inputs):
                continue
            others = set()
            for other_name, _, other_triggers in registered:
                if other_name != name:
                    others.update(other_triggers)
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
        "modalities": {"in": modalities_in or ["text"], "out": outputs},
        "tasks": tasks,
        "capabilities": list(capabilities),
    }
    if "adapters" in capabilities:
        manifest["adapters"] = {"lora": {"targets": list(getattr(pipeline_cls, "_lora_loadable_modules", []))}}
    return manifest


def _workflow_tasks(
    workflow_map: dict[str, Any], inputs: dict[str, dict[str, Any]], output: str
) -> list[tuple[str, tuple[str, ...], set[str]]]:
    """One `(task, required inputs, trigger inputs)` per workflow whose triggers are all inputs. A workflow is named
    after the media inputs it requires when that is a standard task, and keeps its own name otherwise."""
    names = {
        (): f"text_to_{output}",
        ("image",): f"image_to_{output}",
        ("video",): f"video_to_{output}",
        ("image", "mask"): "inpaint",
    }
    tasks = []
    for workflow, spellings in workflow_map.items():
        # A tuple lists alternative spellings of one workflow, e.g. `image` or `last_image`. The first names it and
        # sets what it requires; the inputs of every spelling are left out of the other tasks, which would run it.
        spellings = spellings if isinstance(spellings, tuple) else (spellings,)
        triggers = spellings[0]
        required = tuple(sorted("mask" if key == "mask_image" else key for key in triggers if key != "prompt"))
        if any(value is not True for value in triggers.values()) or not set(required) <= set(inputs):
            logger.warning(f"Not serving workflow {workflow!r}: its triggers {triggers} are not all inputs.")
            continue
        name = names.get(required, workflow)
        if any(name == task[0] for task in tasks):
            logger.warning(f"Not serving workflow {workflow!r}: another workflow is already served as {name!r}.")
            continue
        all_triggers = {"mask" if key == "mask_image" else key for spelling in spellings for key in spelling}
        tasks.append((name, required, all_triggers & set(inputs)))
    return tasks


def _parameter_schema(annotation: Any) -> dict[str, Any] | None:
    """The GMS schema of a scalar annotation, a list of scalars, or a union of them; None for anything else."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        # Any member's values are valid, so members that aren't scalars or lists of them, like a tensor, are left out.
        options = [schema for schema in map(_parameter_schema, typing.get_args(annotation)) if schema is not None]
        # `str | list[str]` is a scalar that also accepts a batch; only a bare list is an array parameter.
        options = [option for option in options if option["type"] != "array"] or options
        order = list(_SCALAR_TYPES.values())
        return min(options, key=lambda option: order.index(option.get("items", option)["type"]), default=None)
    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        items = _parameter_schema(args[0]) if args else None
        if items is None or items["type"] == "array":
            return None
        return {"type": "array", "items": items}
    if annotation in _SCALAR_TYPES:
        return {"type": _SCALAR_TYPES[annotation]}
    return None


# ---------------------------------------------------------------------------
# The manifest bound to the pipeline: startup checks, discovery and request resolution
# ---------------------------------------------------------------------------


class Manifest:
    """A model's manifest, bound to the pipeline that serves it: discovery and request resolution.

    The manifest describes this server, so everything it declares must be servable: each input and parameter maps to a
    kwarg of the pipeline call, and each capability is one this server provides. Anything else is an error at startup
    rather than a request that fails later.
    """

    def __init__(
        self,
        raw: bytes,
        data: dict[str, Any],
        pipeline_cls: type,
        signature: inspect.Signature,
        supported_capabilities: list[str],
        max_adapters: int,
        source: str | None = None,
        revision: str | None = None,
        output_fields: set[str] | None = None,
    ):
        # The bytes are kept as published: discovery serves them unchanged, and the hash is computed over them.
        self.raw = raw
        self.source = source
        self.id = data["id"]
        self.revision = revision
        self.tasks: dict[str, dict[str, Any]] = data["tasks"]
        self.modalities = data.get("modalities", {})
        self.capabilities: list[str] = data.get("capabilities", [])
        self._declared_capabilities = list(self.capabilities)
        self.adapters = data.get("adapters", {})
        self._kwarg_names = set(signature.parameters) - {"self"}
        self._pipeline_name = pipeline_cls.__name__
        self._max_adapters = max_adapters
        # Only the loader's named options (e.g. `load_into_transformer_ref`), not the Hub kwargs in its `**kwargs`.
        lora_parameters = (
            inspect.signature(pipeline_cls.load_lora_weights).parameters.values()
            if hasattr(pipeline_cls, "load_lora_weights")
            else []
        )
        self._lora_options = {
            parameter.name for parameter in lora_parameters if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        } - {"self", "pretrained_model_name_or_path_or_dict", "adapter_name", "hotswap"}
        fps_kwarg = self._pipeline_kwarg("fps", _PARAMETER_KWARGS, {})
        default_fps = signature.parameters[fps_kwarg].default if fps_kwarg is not None else None
        self._default_fps = None if default_fps is inspect.Parameter.empty else default_fps

        if "adapters" in self.capabilities and "adapters" not in supported_capabilities:
            logger.warning(
                f"The GMS manifest for {self.id!r} declares the `adapters` capability, but request adapters are off: "
                f"{self._pipeline_name} cannot load LoRAs, or the server was started with --compile. Serving without "
                "them."
            )
            self.capabilities = [capability for capability in self.capabilities if capability != "adapters"]
        for capability in self.capabilities:
            if capability not in supported_capabilities:
                raise SystemExit(
                    f"The GMS manifest for {self.id!r} declares the {capability!r} capability, which this server does "
                    f"not provide. It provides: {list(supported_capabilities)}."
                )
        if "adapters" in self.capabilities and not self.adapters:
            raise SystemExit(
                f"The GMS manifest for {self.id!r} declares the `adapters` capability but no top-level `adapters` "
                'section naming the adapter types, e.g. {"lora": {"targets": ["transformer"]}}.'
            )

        # Per task: GMS input or parameter name -> the pipeline kwarg it sets.
        self._input_kwargs: dict[str, dict[str, str]] = {}
        self._parameter_kwargs: dict[str, dict[str, str]] = {}
        # Per task: the pipeline kwarg that sets the frame rate, if the pipeline takes one.
        self._fps_kwargs: dict[str, str | None] = {}
        for name, task in self.tasks.items():
            declared = task.get(EXTENSION_KEY, {})
            self._input_kwargs[name] = {
                key: self._require_kwarg(name, "input", key, _INPUT_KWARGS, declared.get("inputs", {}))
                for key in task["inputs"]
            }
            self._parameter_kwargs[name] = {
                key: self._require_kwarg(name, "parameter", key, _PARAMETER_KWARGS, declared.get("parameters", {}))
                for key in task.get("parameters", {})
            }
            self._fps_kwargs[name] = self._parameter_kwargs[name].get("fps", fps_kwarg)
            unsupported = [spec["type"] for spec in task["outputs"].values() if spec["type"] not in OUTPUT_FORMATS]
            if unsupported:
                raise SystemExit(
                    f"The GMS manifest for {self.id!r} declares {unsupported} outputs on task {name!r}; this server "
                    f"produces {list(OUTPUT_FORMATS)} outputs."
                )
            # `output_fields` is None when the pipeline doesn't annotate what it returns.
            if output_fields is not None:
                for spec in task["outputs"].values():
                    if not set(OUTPUT_FIELDS[spec["type"]]) & output_fields:
                        raise SystemExit(
                            f"Task {name!r} outputs {spec['type']}, read from one of "
                            f"{list(OUTPUT_FIELDS[spec['type']])}, but {self._pipeline_name} returns "
                            f"{sorted(output_fields)}. `diffusers serve` only serves image, video and audio outputs."
                        )
            parameters = task.get("parameters", {})
            for key, spec in parameters.items():
                if "lattice" in spec:
                    logger.warning(
                        f"Ignoring the lattice on parameter {key!r} of task {name!r}: `diffusers serve` passes its "
                        "values to the pipeline as they are."
                    )
            produces_video = any(spec["type"] == "video" for spec in task["outputs"].values())
            if produces_video and self._default_fps is None:
                logger.warning(
                    f"{self._pipeline_name} has no default `fps` or `frame_rate` argument, so task {name!r} encodes "
                    f"videos at {DEFAULT_FPS} fps, which may not be the model's native rate."
                )
            for preset, body in task.get("presets", {}).items():
                for key in body.get("parameters", {}):
                    if key not in parameters:
                        raise SystemExit(
                            f"The GMS manifest for {self.id!r} sets parameter {key!r} in preset {preset!r} of task "
                            f"{name!r}, which the task does not declare."
                        )

    def _pipeline_kwarg(self, name: str, aliases: dict[str, tuple[str, ...]], declared: dict[str, str]) -> str | None:
        candidates = [declared[name]] if name in declared else [*aliases.get(name, ()), name]
        return next((kwarg for kwarg in candidates if kwarg in self._kwarg_names), None)

    def _require_kwarg(
        self, task: str, kind: str, name: str, aliases: dict[str, tuple[str, ...]], declared: dict[str, str]
    ) -> str:
        kwarg = self._pipeline_kwarg(name, aliases, declared)
        if kwarg is None:
            raise SystemExit(
                f"The GMS manifest for {self.id!r} declares {kind} {name!r} on task {task!r}, which "
                f"{self._pipeline_name} does not take. Remove it from the manifest, or map it to a pipeline kwarg "
                f'under `{EXTENSION_KEY}`, e.g. "{EXTENSION_KEY}": {{"{kind}s": {{"{name}": "<kwarg>"}}}}.'
            )
        return kwarg

    def discovery_document(self) -> bytes:
        # The manifest bytes are spliced in untouched: the spec defines verification as a hash of the served bytes.
        document = '{"spec": ' + json.dumps(SPEC_VERSION)
        if self.source is not None:
            document += ', "source": ' + json.dumps(self.source)
            document += f', "hash": "sha256:{hashlib.sha256(self.raw).hexdigest()}"'
        # A server that provides less than the manifest declares says so in the spec's `server` overlay.
        if self.capabilities != self._declared_capabilities:
            document += ', "server": ' + json.dumps({"capabilities": self.capabilities})
        document += ', "manifest": '
        return document.encode() + self.raw + b"}"

    def resolve(self, body: Any) -> ResolvedRequest:
        """Validate a create-generation request body and map it to the pipeline call."""
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

        # The task: the one requested, else the only one whose inputs match what was supplied.
        task = body.get("task")
        if task is not None:
            if not isinstance(task, str) or task not in self.tasks:
                raise GMSError(
                    "invalid_request",
                    f"task {task!r} is not hosted for {self.id!r}; hosted tasks: {list(self.tasks)}",
                    "/task",
                )
        elif len(self.tasks) == 1:
            task = next(iter(self.tasks))
        else:
            candidates = []
            for name, spec in self.tasks.items():
                required = {key for key, input_spec in spec["inputs"].items() if input_spec.get("required")}
                if required <= set(inputs) <= set(spec["inputs"]):
                    candidates.append(name)
            if len(candidates) == 1:
                task = candidates[0]
            else:
                raise GMSError(
                    "invalid_request",
                    f"could not infer the task from inputs {sorted(inputs)}; set `task` to one of {list(self.tasks)}",
                    "/task",
                )
        declared_inputs = self.tasks[task]["inputs"]
        declared_parameters = self.tasks[task].get("parameters", {})

        # The parameter values: declared defaults, then the preset, then the request's own. Values the request set
        # itself are held to stricter rules than defaults.
        explicit = body.get("parameters", {})
        if not isinstance(explicit, dict):
            raise GMSError("invalid_request", "parameters must be an object", "/parameters")
        for key, value in explicit.items():
            if key not in declared_parameters:
                raise GMSError("invalid_request", f"task {task!r} declares no parameter {key!r}", f"/parameters/{key}")
            _check_value(key, value, declared_parameters[key], f"/parameters/{key}")
        values = {key: spec["default"] for key, spec in declared_parameters.items() if "default" in spec}
        preset = body.get("preset")
        if preset is not None:
            presets = self.tasks[task].get("presets", {})
            if preset not in presets:
                raise GMSError(
                    "invalid_request",
                    f"preset {preset!r} is not declared for task {task!r}; declared presets: {list(presets)}",
                    "/preset",
                )
            values.update(presets[preset].get("parameters", {}))
        values.update(explicit)

        # Inputs: text is passed as is; media is checked here and loaded by the backend.
        kwargs: dict[str, Any] = {}
        media: dict[str, tuple[str, str, Any]] = {}
        for key, value in inputs.items():
            pointer = f"/inputs/{key}"
            spec = declared_inputs.get(key)
            if spec is None:
                raise GMSError("invalid_request", f"task {task!r} declares no input {key!r}", pointer)
            kwarg = self._input_kwargs[task][key]
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
            media[kwarg] = (key, spec["type"], value)
        for key, spec in declared_inputs.items():
            if spec.get("required") and key not in inputs:
                raise GMSError("invalid_request", f"task {task!r} requires input {key!r}", f"/inputs/{key}")

        # Parameters. The seed is pulled out, since it seeds a generator instead of being passed as is.
        seed = None
        for key, value in values.items():
            spec = declared_parameters[key]
            kwarg = self._parameter_kwargs[task][key]
            pointer = f"/parameters/{key}"
            if "active_when" in spec and not _is_active(spec["active_when"], values):
                if key in explicit:
                    raise GMSError(
                        "invalid_request",
                        f"parameter {key!r} is inactive unless {json.dumps(spec['active_when'])}",
                        pointer,
                    )
                continue
            if key == "seed":
                if not 0 <= value < 2**63:
                    raise GMSError("invalid_request", f"seed must be between 0 and 2**63 - 1, got {value}", pointer)
                seed = value
                continue
            kwargs[kwarg] = value
        # The spec has the server pick the seed when the request omits it, so the result is reproducible.
        if seed is None and "seed" in self._parameter_kwargs[task]:
            seed = random.randrange(2**32)

        # Outputs, each with the media type it is encoded as.
        output_formats = {kind: formats[0] for kind, formats in OUTPUT_FORMATS.items()}
        output_format = body.get("output_format")
        if output_format is not None:
            kind = output_format.split("/")[0] if isinstance(output_format, str) else None
            produced = {spec["type"] for spec in self.tasks[task]["outputs"].values()}
            # The startup check limits output types to those in OUTPUT_FORMATS.
            if kind not in produced or output_format not in OUTPUT_FORMATS[kind]:
                supported = sorted(
                    media_type for produced_kind in produced for media_type in OUTPUT_FORMATS[produced_kind]
                )
                raise GMSError(
                    "invalid_request",
                    f"output_format must be one of {supported} for task {task!r}, got {output_format!r}",
                    "/output_format",
                )
            output_formats[kind] = output_format
        outputs = {
            name: (spec["type"], output_formats[spec["type"]]) for name, spec in self.tasks[task]["outputs"].items()
        }
        fps = None
        if any(kind == "video" for kind, _ in outputs.values()):
            fps = kwargs.get(self._fps_kwargs[task]) or self._default_fps or DEFAULT_FPS

        response_format = body.get("response_format", "url")
        if response_format not in ("url", "b64_json"):
            raise GMSError(
                "invalid_request",
                f"response_format must be 'url' or 'b64_json', got {response_format!r}",
                "/response_format",
            )

        # Adapters, each resolved to its path and the weight `set_adapters` takes.
        adapters = []
        if body.get("adapters") is not None:
            if "adapters" not in self.capabilities:
                raise GMSError("not_implemented", "this server does not apply adapters", "/adapters")
            if not isinstance(body["adapters"], list):
                raise GMSError("invalid_request", "adapters must be an array", "/adapters")
            if len(body["adapters"]) > self._max_adapters:
                raise GMSError(
                    "invalid_request",
                    f"at most {self._max_adapters} adapters per request, got {len(body['adapters'])}",
                    "/adapters",
                )
            for index, adapter in enumerate(body["adapters"]):
                pointer = f"/adapters/{index}"
                if not isinstance(adapter, dict):
                    raise GMSError("invalid_request", "an adapter must be an object", pointer)
                for key in adapter:
                    if key not in _ADAPTER_FIELDS and key not in self._lora_options:
                        raise GMSError(
                            "invalid_request",
                            f"unknown adapter field {key!r}; adapters accept "
                            f"{sorted({*_ADAPTER_FIELDS, *self._lora_options})}",
                            f"{pointer}/{key}",
                        )
                if not isinstance(adapter.get("type"), str) or adapter["type"] not in self.adapters:
                    raise GMSError(
                        "invalid_request",
                        f"adapter type must be one of {list(self.adapters)}, got {adapter.get('type')!r}",
                        f"{pointer}/type",
                    )
                path = adapter.get("path")
                repo_id = path.removeprefix("hf:") if isinstance(path, str) and path.startswith("hf:") else None
                if repo_id is not None:
                    try:
                        validate_repo_id(repo_id)
                    except HFValidationError:
                        repo_id = None
                # `load_lora_weights` prefers a local path over a Hub repo of the same name, so a repo id that also
                # names a path on the server's disk is refused as well.
                if repo_id is None or "/" not in repo_id or os.path.exists(repo_id):
                    raise GMSError(
                        "invalid_request",
                        f"adapter path must be a Hub repo id of the form 'hf:<namespace>/<name>', got {path!r}",
                        f"{pointer}/path",
                    )
                scale = adapter.get("scale", 1.0)
                if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                    raise GMSError(
                        "invalid_request", f"adapter scale must be a number, got {scale!r}", f"{pointer}/scale"
                    )
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
                weight = {target: targets.get(target, scale) for target in valid_targets} if targets else scale
                options = {key: value for key, value in adapter.items() if key in self._lora_options}
                adapters.append({"path": path, "weight": weight, "options": options})

        return ResolvedRequest(
            task=task,
            kwargs=kwargs,
            media=media,
            seed=seed,
            outputs=outputs,
            fps=fps,
            response_format=response_format,
            adapters=adapters,
        )


# ---------------------------------------------------------------------------
# Value checks
# ---------------------------------------------------------------------------


def _check_value(label: str, value: Any, spec: dict[str, Any], pointer: str) -> None:
    """Check a parameter value against its declared schema: type, enum, range and `multiple_of`."""
    kind = spec.get("type")
    if kind == "array":
        if not isinstance(value, list):
            raise GMSError("invalid_request", f"{label} must be an array, got {value!r}", pointer)
        for index, item in enumerate(value):
            _check_value(f"{label}[{index}]", item, spec["items"], f"{pointer}/{index}")
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
            f"{name} must be either {{type, url, media_type?}} or {{type, base64, media_type}}, "
            f"got keys {sorted(keys)}",
            pointer,
        )
    # A URL media value is fetched by the server, so anything but http(s) would read the server's own filesystem.
    if not isinstance(value["url"], str) or not value["url"].startswith(("http://", "https://")):
        raise GMSError("invalid_request", f"{name} url must be an http(s) URL", f"{pointer}/url")


def _is_active(condition: dict[str, Any], values: dict[str, Any]) -> bool:
    """Evaluate an `active_when` condition against the parameter values."""
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
