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

"""Compose source files and unpack tensor trees before applying a component conversion."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch


class MergedCheckpoint(Mapping):
    """Merge disjoint tensor dictionaries lazily, with optional component namespaces."""

    def __init__(self, sources):
        self.sources = tuple(sources)
        self.locations = {}
        for index, (prefix, source) in enumerate(self.sources):
            for key in source:
                target = prefix + key
                if target in self.locations:
                    raise ValueError(f"Duplicate source tensor: {target}.")
                self.locations[target] = (index, key)
        if not self.locations:
            raise ValueError("No tensors found in the checkpoint sources.")

    def __getitem__(self, key):
        index, source_key = self.locations[key]
        return self.sources[index][1][source_key]

    def __iter__(self):
        return iter(self.locations)

    def __len__(self):
        return len(self.locations)


def load_source_manifest(path):
    """Load JSON `sources` entries containing path, input_prefix, output_prefix and wrapper fields.

    Paths are local and relative to the manifest directory unless absolute. Prefixes select or namespace complete
    components; they do not rename model parameters. Tensor conversion remains the registered `Conversion`.
    """
    from .io import Checkpoint

    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"sources"} or not isinstance(manifest["sources"], list):
        raise ValueError("A source manifest must contain a sources list.")
    sources = []
    for entry in manifest["sources"]:
        allowed = {"path", "input_prefix", "output_prefix", "wrapper", "format"}
        if not isinstance(entry, dict) or "path" not in entry or set(entry) - allowed:
            raise ValueError("Source entries require path and may specify input_prefix, output_prefix and wrapper.")
        for key in ("path", "input_prefix", "output_prefix"):
            if not isinstance(entry.get(key, ""), str):
                raise ValueError(f"Source {key} must be a string.")
        wrapper = entry.get("wrapper", [])
        if not isinstance(wrapper, list) or not all(isinstance(key, str) for key in wrapper):
            raise ValueError("Source wrapper must be a list of nested dictionary keys.")
        source_path = path.parent / entry["path"]
        source_format = entry.get("format", "tensors")
        if source_format in ("torchscript", "python-model"):
            prefix = entry.get("input_prefix", "")
            if source_format == "torchscript":
                if wrapper:
                    raise ValueError("TorchScript sources do not use dictionary wrappers.")
                model = torch.jit.load(str(source_path), map_location="cpu")
            else:
                # Explicit opt-in for legacy research checkpoints containing a pickled nn.Module.
                # Loading these executes pickle code and requires the original Python model package.
                model = torch.load(source_path, map_location="cpu", weights_only=False)
                for key in wrapper:
                    model = model[key]
            state = model.state_dict()
            checkpoint = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        elif source_format == "tensors":
            checkpoint = Checkpoint(source_path, prefix=entry.get("input_prefix", ""), wrapper=wrapper)
        else:
            raise ValueError("Source format must be tensors, torchscript or python-model.")
        sources.append((entry.get("output_prefix", ""), checkpoint))
    return MergedCheckpoint(sources)


def flatten_tensor_tree(weights, separator="."):
    """Flatten nested original NumPy/JAX tensor dictionaries without changing tensor axes or values."""
    result = {}

    def visit(tree, prefix):
        for name, value in tree.items():
            key = prefix + str(name)
            if isinstance(value, Mapping):
                visit(value, key + separator)
            else:
                if key in result:
                    raise ValueError(f"Duplicate flattened tensor: {key}.")
                result[key] = torch.as_tensor(value)

    visit(weights, "")
    return result


def load_tensor_sources(path):
    """Read a tensor file, indexed component directory, or directory of disjoint safetensors shards."""
    from .io import Checkpoint

    path = Path(path)
    if not path.is_dir():
        return Checkpoint(path)
    if any(path.glob("*.index.json")) or any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
        )
    ):
        return Checkpoint(path)
    return MergedCheckpoint(("", Checkpoint(shard)) for shard in sorted(path.glob("*.safetensors")))
