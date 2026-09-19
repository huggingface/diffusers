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

"""Reusable original configuration preparation, separate from the tensor conversion definitions."""

import copy
import importlib
import json
from collections.abc import Mapping
from pathlib import Path


def list_config_presets():
    directory = Path(__file__).resolve().parent
    catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
    return sorted(
        [path.stem for path in (directory.parent / "presets").glob("*.json")]
        + [f"{module}.{name}" for module, names in catalog.items() for name in names]
    )


def get_config_preset(name, *, arguments=None, component=None):
    """Load a built-in config or call a named original-config helper with explicit keyword arguments.

    `component` selects an entry in a collection, such as Cosmos' `TRANSFORMER_CONFIGS`. Helpers returning original
    download metadata expose their `diffusers_config` field automatically. Config preparation never authors tensor
    mappings; pass the resulting dictionary to `get_conversion` or the checkpoint CLI.
    """
    if name not in list_config_presets():
        raise ValueError(f"Unknown config preset {name!r}; use --list-presets to see available names.")
    directory = Path(__file__).resolve().parent
    arguments = dict(arguments or {})
    if "." in name and not (directory.parent / "presets" / (name + ".json")).is_file():
        module, attribute = name.split(".", 1)
        value = getattr(importlib.import_module(f"{__name__}.{module}"), attribute)
        if callable(value):
            value = value(**arguments)
        elif arguments:
            raise ValueError("Dictionary presets do not take arguments; use component to select a variant.")
    else:
        if arguments:
            raise ValueError("JSON presets do not take arguments.")
        value = json.loads((directory.parent / "presets" / (name + ".json")).read_text(encoding="utf-8"))
    if component is not None:
        value = value[component]
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("The selected preset did not produce a configuration dictionary.")
    return copy.deepcopy(dict(value.get("diffusers_config", value)))
