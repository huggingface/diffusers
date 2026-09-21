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

"""Local tensor checkpoint I/O, separate from the reversible component definitions."""

import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .registry import get_conversion


class Checkpoint(Mapping[str, Any]):
    """Read a component checkpoint lazily, retaining at most one deserialized PyTorch shard.

    Safetensors are memory mapped per access. PyTorch files are loaded on CPU with `weights_only=True`. An explicit
    wrapper selects a nested state dict; a prefix selects one component without discarding unexpected keys inside it.
    Safe auxiliary metadata is retained until component preparation; tensor rules validate the values they consume.
    Indexes must describe every tensor in every referenced shard exactly once, with paths inside the index directory.
    """

    def __init__(self, path: str | Path, *, prefix: str = "", wrapper: Sequence[str] = ()) -> None:
        path = Path(path)
        self.prefix, self.wrapper = prefix, tuple(wrapper)
        self._aliases = {}
        self._cached_path, self._cached_state = None, None
        if path.is_dir():
            names = (
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.safetensors.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "diffusion_pytorch_model.bin",
                "diffusion_pytorch_model.bin.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            path = next((path / name for name in names if (path / name).is_file()), None)
            if path is None:
                raise FileNotFoundError("No tensor checkpoint or shard index found in the component directory.")
        if path.name.endswith(".index.json"):
            index = json.loads(path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("A checkpoint index needs a nonempty weight_map.")
            root = path.parent.resolve()
            paths = {}
            for name in set(weight_map.values()):
                shard = (root / name).resolve()
                if Path(name).is_absolute() or not shard.is_relative_to(root):
                    raise ValueError(f"Checkpoint shard escapes the index directory: {name}.")
                paths[name] = shard
            actual = {}
            for name, shard in paths.items():
                for key in self._keys(shard):
                    if key in actual:
                        raise ValueError(f"Duplicate key across checkpoint shards: {key}.")
                    actual[key] = name
            if actual != weight_map:
                raise ValueError("Checkpoint index does not match the keys and locations in its shards.")
            self._paths = {key: paths[name] for key, name in weight_map.items()}
        else:
            self._paths = dict.fromkeys(self._keys(path), path)
        self._paths = {key[len(prefix) :]: path for key, path in self._paths.items() if key.startswith(prefix)}
        if not self._paths:
            raise ValueError(f"No tensor keys match prefix {prefix!r}.")

    def _load(self, path: Path) -> Mapping[str, Any]:
        if self._cached_path != path:
            state = torch.load(path, map_location="cpu", weights_only=True)
            for key in self.wrapper:
                if not isinstance(state, Mapping) or key not in state:
                    raise ValueError(f"Missing checkpoint wrapper {self.wrapper} in {path}.")
                state = state[key]
            if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
                raise ValueError("Expected a string-keyed state dict; select nested weights with wrapper.")
            if not any(isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("Expected a tensor state dict; select nested weights with wrapper.")
            self._cached_path, self._cached_state = path, state
        return self._cached_state

    def _keys(self, path: Path) -> Sequence[str]:
        if path.suffix == ".safetensors":
            if self.wrapper:
                raise ValueError("Safetensors do not have nested checkpoint wrappers.")
            with safe_open(path, framework="pt", device="cpu") as handle:
                return handle.keys()
        return tuple(self._load(path))

    def __getitem__(self, key: str) -> Any:
        key = self._aliases.get(key, key)
        path = self._paths[key]
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(self.prefix + key)
        return self._load(path)[self.prefix + key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def convert_checkpoint(
    input_path: str | Path | Mapping[str, Any],
    output_path: str | Path,
    *,
    config: dict[str, Any],
    model_class: str | None = None,
    reverse: bool = False,
    input_prefix: str = "",
    input_wrapper: Sequence[str] = (),
    output_prefix: str = "",
    output_format: Literal["safetensors", "pytorch"] = "safetensors",
    output_wrapper: Sequence[str] = (),
    max_shard_size: int = 5_000_000_000,
) -> Path:
    """Write a new safetensors component directory or a single PyTorch checkpoint.

    `config` is the matching Diffusers component configuration in either direction. Original inputs must use the
    selected definition's canonical component layout. The output contains safetensors plus configuration/format
    metadata; original runtime configuration, pipeline assets and training state are not reconstructed. Existing output
    directories are never overwritten. Conversion failures leave no partially published output directory.
    """
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive.")
    if output_format not in ("safetensors", "pytorch"):
        raise ValueError("output_format must be safetensors or pytorch.")
    if output_wrapper and output_format != "pytorch":
        raise ValueError("Nested output wrappers require PyTorch serialization.")
    if config.get("quantization_config") is not None:
        raise ValueError("Conversion requires unpacked, unquantized tensor weights.")
    model_class = model_class or config.get("_class_name")
    if model_class is None and len(config.get("architectures", [])) == 1:
        model_class = config["architectures"][0]
    conversion = get_conversion(model_class, config)
    if isinstance(input_path, Mapping):
        if input_prefix or input_wrapper:
            raise ValueError("Select prefixes and wrappers when composing the input tensor mapping.")
        checkpoint = input_path
    else:
        checkpoint = Checkpoint(input_path, prefix=input_prefix, wrapper=input_wrapper)
    if not reverse and model_class == "CogVideoXTransformer3DModel":
        from .checkpoint import ComponentState
        from .cogvideox import cogvideox_transformer_auxiliary_keys

        checkpoint = ComponentState(checkpoint)
        for key in cogvideox_transformer_auxiliary_keys(config):
            checkpoint.keys_to_source.pop(key, None)
    elif not reverse and model_class == "AutoencoderKLCogVideoX":
        from .checkpoint import ComponentState

        checkpoint = ComponentState(checkpoint)
        checkpoint.keys_to_source = {
            key: value for key, value in checkpoint.keys_to_source.items() if not key.startswith("loss.")
        }
    # Transformers omits duplicate storage when saving tied embeddings as safetensors. Restore only declared ties,
    # never infer an arbitrary missing parameter from another tensor with the same shape.
    tied_groups = []
    if model_class in ("T5EncoderModel", "UMT5EncoderModel"):
        tied_groups.append(("shared.weight", "encoder.embed_tokens.weight"))
    if model_class == "Qwen3ForCausalLM" and config.get("tie_word_embeddings", False):
        tied_groups.append(("model.embed_tokens.weight", "lm_head.weight"))
    for group in tied_groups:
        source = next((key for key in group if key in checkpoint), None)
        if source is not None:
            for key in group:
                if key not in checkpoint:
                    if isinstance(checkpoint, Checkpoint):
                        checkpoint._aliases[key] = source
                        checkpoint._paths[key] = checkpoint._paths[source]
                    else:
                        from .checkpoint import ComponentState

                        if not isinstance(checkpoint, ComponentState):
                            checkpoint = ComponentState(checkpoint)
                        checkpoint.keys_to_source[key] = source
    if not reverse and model_class not in ("CogVideoXTransformer3DModel", "AutoencoderKLCogVideoX"):
        from .checkpoint import prepare_component_checkpoint

        checkpoint, conversion, config = prepare_component_checkpoint(checkpoint, config, model_class)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".conversion-", dir=output_path.parent) as temporary:
        if output_format == "pytorch":
            converted = dict(conversion.iter_converted(checkpoint, reverse=reverse))
            if (
                reverse
                and model_class == "CogVideoXTransformer3DModel"
                and not config.get("use_rotary_positional_embeddings", False)
            ):
                from .cogvideox import cogvideox_fixed_position_embedding

                converted["mixins.pos_embed.pos_embedding"] = cogvideox_fixed_position_embedding(
                    config,
                    checkpoint["patch_embed.proj.weight"].dtype,
                )
            converted = {output_prefix + key: tensor for key, tensor in converted.items()}
            for wrapper in reversed(output_wrapper):
                converted = {wrapper: converted}
            staging = Path(temporary) / "checkpoint.pt"
            torch.save(converted, staging)
            staging.rename(output_path)
            return output_path
        staging = Path(temporary) / "component"
        staging.mkdir()
        shards, current, size, total_size = [], {}, 0, 0

        def flush():
            if not current:
                return
            filename = f"part-{len(shards) + 1:05d}.safetensors"
            save_file(current, staging / filename, metadata={"format": "pt"})
            shards.append((filename, tuple(current)))
            current.clear()

        for key, tensor in conversion.iter_converted(checkpoint, reverse=reverse):
            tensor_size = tensor.numel() * tensor.element_size()
            if size and size + tensor_size > max_shard_size:
                flush()
                size = 0
            # Split views and tied parameters must have independent contiguous storage for safetensors.
            current[output_prefix + key] = tensor.detach().cpu().contiguous().clone()
            size += tensor_size
            total_size += tensor_size
        flush()
        base = "model" if reverse or config.get("model_type") else "diffusion_pytorch_model"
        weight_map = {}
        for i, (filename, keys) in enumerate(shards, 1):
            final_name = (
                f"{base}.safetensors" if len(shards) == 1 else f"{base}-{i:05d}-of-{len(shards):05d}.safetensors"
            )
            (staging / filename).rename(staging / final_name)
            weight_map.update(dict.fromkeys(keys, final_name))
        if len(shards) > 1:
            (staging / f"{base}.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n",
                encoding="utf-8",
            )
        saved_config = dict(config, _class_name=model_class)
        config_name = "conversion_config.json" if reverse else "config.json"
        (staging / config_name).write_text(json.dumps(saved_config, indent=2) + "\n", encoding="utf-8")
        (staging / "conversion.json").write_text(
            json.dumps(
                {
                    "model_class": model_class,
                    "direction": "original" if reverse else "diffusers",
                    "lossless": conversion.lossless,
                    "output_prefix": output_prefix,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(output_path)
    return output_path
