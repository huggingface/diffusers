# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Shared utilities for transformer model implementations."""

from dataclasses import dataclass, fields

import torch


@dataclass
class TransformerModuleOutput:
    """Base class providing tuple-compatible iteration for structured submodule outputs.

    Doesn't declare any fields itself — subclasses define their own schema. Provides only the plumbing that lets
    callers unpack positionally (``h, e = output``), index (``output[0]``), and check length, with ``None`` fields
    transparently skipped so a single-stream output unpacks as a 1-tuple. This matches the legacy bare-tuple return
    shape so subclasses can be adopted without touching callers.
    """

    def _as_tuple(self):
        """Tuple-compat view of the dataclass: declared field order, with ``None`` values skipped."""
        return tuple(getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None)

    def __iter__(self):
        return iter(self._as_tuple())

    def __getitem__(self, idx):
        return self._as_tuple()[idx]

    def __len__(self):
        return len(self._as_tuple())


@dataclass
class TransformerBlockOutput(TransformerModuleOutput):
    """Structured return type for transformer-block ``forward`` methods.

    Replaces the historical pattern of returning bare tuples whose element ordering varied per model (e.g. Flux
    returned ``(encoder_hidden_states, hidden_states)`` while CogVideoX returned ``(hidden_states,
    encoder_hidden_states)``). Tuple-compatibility inherited from :class:`TransformerModuleOutput`.

    Attributes:
        hidden_states: The block's primary output tensor. Always populated.
        encoder_hidden_states: The text / context stream output for dual-stream blocks. ``None`` for single-stream.
    """

    hidden_states: torch.Tensor = None
    encoder_hidden_states: torch.Tensor | None = None


@dataclass
class AttnProcessorOutput(TransformerModuleOutput):
    """Structured return type for attention-processor ``__call__`` methods.

    Replaces the historical pattern of returning a bare tensor for single-stream attention and a bare
    ``(hidden_states, encoder_hidden_states)`` tuple for dual-stream attention. Tuple-compatibility inherited from
    :class:`TransformerModuleOutput`.

    Attributes:
        hidden_states: The processor's primary output tensor. Always populated.
        encoder_hidden_states: The text / context stream output for dual-stream attention processors. ``None`` for
            single-stream.
    """

    hidden_states: torch.Tensor = None
    encoder_hidden_states: torch.Tensor | None = None
