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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from .transforms import Identity, Transform


@dataclass(frozen=True)
class Rule:
    """Map an ordered group of original keys to Diffusers keys through a reversible tensor operation."""

    original: tuple[str, ...]
    diffusers: tuple[str, ...]
    transform: Transform = field(default_factory=Identity)

    def __post_init__(self):
        for name in ("original", "diffusers"):
            keys = getattr(self, name)
            if isinstance(keys, str) or not keys:
                raise ValueError(f"Rule.{name} must be a nonempty sequence of keys, not a string.")
            keys = tuple(keys)
            object.__setattr__(self, name, keys)
            if not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError(f"Rule.{name} keys must be nonempty strings.")
            if len(set(keys)) != len(keys):
                raise ValueError(f"Repeated {name} key in rule: {keys}.")


@dataclass(frozen=True)
class Conversion:
    """Convert a component state dict in either direction using a single definition.

    `mapping` is shorthand for exact one-to-one renames; `rules` describe grouped tensor operations. Every source key
    must be consumed once and every destination key written once. Known auxiliary keys and component prefixes must be
    handled before conversion. Both methods leave the input dictionary and tensors unchanged, but outputs may share
    storage with inputs. No model construction, file I/O, device movement, or dtype conversion is performed here.
    """

    mapping: Mapping[str, str] = field(default_factory=dict)
    rules: tuple[Rule, ...] = ()
    original_keys: frozenset[str] = field(init=False)
    diffusers_keys: frozenset[str] = field(init=False)
    _rules: tuple[Rule, ...] = field(init=False, repr=False)
    lossless: bool = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))
        object.__setattr__(self, "rules", tuple(self.rules))
        rules = tuple(Rule((old,), (new,)) for old, new in self.mapping.items()) + self.rules
        original_keys, diffusers_keys = set(), set()
        for rule in rules:
            repeated_inputs = original_keys.intersection(rule.original)
            repeated_outputs = diffusers_keys.intersection(rule.diffusers)
            if repeated_inputs or repeated_outputs:
                raise ValueError(
                    f"Conversion keys must be unique: repeated original keys {sorted(repeated_inputs)}, "
                    f"repeated Diffusers keys {sorted(repeated_outputs)}."
                )
            original_keys.update(rule.original)
            diffusers_keys.update(rule.diffusers)
        object.__setattr__(self, "original_keys", frozenset(original_keys))
        object.__setattr__(self, "diffusers_keys", frozenset(diffusers_keys))
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "lossless", all(getattr(rule.transform, "lossless", True) for rule in rules))

    def to_diffusers(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert original component weights to Diffusers weights without modifying the inputs."""
        return self._convert(state_dict, reverse=False)

    def to_original(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert Diffusers component weights to original weights without modifying the inputs."""
        return self._convert(state_dict, reverse=True)

    def _convert(self, state_dict: Mapping[str, torch.Tensor], reverse: bool) -> dict[str, torch.Tensor]:
        return dict(self.iter_converted(state_dict, reverse=reverse))

    def iter_converted(self, state_dict: Mapping[str, torch.Tensor], *, reverse: bool = False):
        """Yield converted key/tensor pairs one rule at a time, after validating complete input coverage.

        File readers can supply a lazy mapping and writers can flush shards between rules. A grouped operation still
        needs all of its input and output tensors in memory. As with the dictionary methods, tensors may alias inputs.
        """
        source_keys = self.diffusers_keys if reverse else self.original_keys
        missing = sorted(source_keys - state_dict.keys())
        unexpected = sorted(state_dict.keys() - source_keys)
        direction = "Diffusers -> original" if reverse else "original -> Diffusers"
        if missing or unexpected:
            raise ValueError(f"Cannot convert {direction}: missing keys {missing}; unexpected keys {unexpected}.")

        for rule in self._rules:
            source = rule.diffusers if reverse else rule.original
            target = rule.original if reverse else rule.diffusers
            transform = rule.transform.inverse if reverse else rule.transform.forward
            try:
                tensors = tuple(state_dict[key] for key in source)
                if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
                    raise ValueError("Every converted parameter must be a tensor.")
                outputs = transform(tensors)
                if len(outputs) != len(target):
                    raise ValueError(f"Expected {len(target)} output tensors, got {len(outputs)}.")
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"Failed to convert {direction}, {source} -> {target}: {error}") from error
            yield from zip(target, outputs)
