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
"""Shared helpers used by multiple ``diffusers-cli`` subcommands.

Anything imported by more than one command file lives here so command modules stay standalone — no cross-command
imports between e.g. ``describe`` and ``generate``.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def try_fetch_config(args: Namespace, filename: str) -> str | None:
    """Resolve ``filename`` for ``args.model`` (local path or Hub repo). Return None if absent.

    Used by ``generate`` (to detect modular vs standard pipelines) and ``describe`` (to read the pipeline class for
    schema introspection) — no weights are downloaded, only the small index file.
    """
    local = Path(args.model)
    if local.exists():
        candidate = local / filename
        return str(candidate) if candidate.exists() else None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    try:
        return hf_hub_download(args.model, filename, revision=args.revision, token=args.token)
    except (EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError):
        return None
