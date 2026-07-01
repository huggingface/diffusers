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
"""``diffusers-cli skills`` — install agent skill bundles into multiple AI coding agents.

Skill bundles live under ``.ai/skills/<name>/`` in the diffusers repo. ``install`` fetches a bundle
via the GitHub Contents API and writes it in the format expected by each target agent:

- ``claude``  → ``.claude/skills/<name>/`` (bundle copied verbatim)
- ``cursor``  → ``.cursor/rules/<name>.mdc`` (frontmatter rewritten for Cursor)
- ``agents-md`` → ``AGENTS.md`` section (covers Codex, Aider, etc.)

Default is auto-detect via env vars set by known agents (``CLAUDECODE``, ``CURSOR_AI``,
``CODEX_SANDBOX``, ``AIDER_AI_CONTEXT``). If no agent is detected, writes to all three so the
bundle is available no matter which agent the user later switches to. Override with
``--agents claude,cursor`` etc.
"""

from __future__ import annotations

import os
import re
import shutil
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

import httpx
from huggingface_hub.cli._output import out

from . import BaseDiffusersCLICommand


_REGISTRY_BASE = "https://api.github.com/repos/huggingface/diffusers/contents/.ai/skills"
_REGISTRY_REF = "main"

# Env var → target name. Matches hub's `is_agent()` detection list.
_AGENT_ENV_TO_TARGET = {
    "CLAUDECODE": "claude",
    "CLAUDE_CODE": "claude",
    "CURSOR_AI": "cursor",
    "CODEX_SANDBOX": "agents-md",
    "AIDER_AI_CONTEXT": "agents-md",
}
_ALL_TARGETS = ("claude", "cursor", "agents-md")


# ---------------------------------------------------------------------------
# Registry fetch
# ---------------------------------------------------------------------------


def _registry_url(name: str = "") -> str:
    """API URL for the registry root, or for a single skill bundle when ``name`` is given."""
    path = f"/{name}" if name else ""
    return f"{_REGISTRY_BASE}{path}?ref={_REGISTRY_REF}"


def _fetch_json(url: str) -> list[dict]:
    try:
        resp = httpx.get(url)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise SystemExit(f"Not found in registry: {url}") from e
        raise SystemExit(f"Registry fetch failed: HTTP {e.response.status_code} {e.response.reason_phrase}") from e
    except httpx.HTTPError as e:
        raise SystemExit(f"Could not reach registry: {e}") from e


def _walk_skill_files(name: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []

    def _walk(api_url: str, prefix: str) -> None:
        for entry in _fetch_json(api_url):
            if entry["type"] == "file":
                files.append((f"{prefix}{entry['name']}", entry["download_url"]))
            elif entry["type"] == "dir":
                _walk(entry["url"], f"{prefix}{entry['name']}/")

    _walk(_registry_url(name), "")
    return files


def _download_bundle(name: str) -> dict[str, bytes]:
    files = _walk_skill_files(name)
    if not files:
        raise SystemExit(f"Skill '{name}' has no files in the registry.")
    bundle: dict[str, bytes] = {}
    for rel_path, url in files:
        resp = httpx.get(url)
        resp.raise_for_status()
        bundle[rel_path] = resp.content
    return bundle


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a Markdown file with YAML frontmatter into ``(metadata, body)``."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    fm_text = text[4:end]
    body = text[end + 5 :]

    metadata: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for raw in fm_text.splitlines():
        if current_key is not None and (raw.startswith("  ") or raw == ""):
            if raw.strip():
                current_lines.append(raw.strip())
            continue
        if current_key is not None:
            metadata[current_key] = " ".join(current_lines).strip()
            current_key, current_lines = None, []
        match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)$", raw)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in ("", ">", "|", ">-", "|-"):
            current_key = key
        else:
            metadata[key] = value
    if current_key is not None:
        metadata[current_key] = " ".join(current_lines).strip()
    return metadata, body


# ---------------------------------------------------------------------------
# Per-agent install adapters
# ---------------------------------------------------------------------------


def _install_claude(name: str, bundle: dict[str, bytes], root: Path, force: bool) -> Path:
    skill_dir = root / ".claude" / "skills" / name
    if skill_dir.exists():
        if not force:
            raise SystemExit(f"Skill already installed at {skill_dir}. Use --force to reinstall.")
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in bundle.items():
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return skill_dir


def _install_cursor(name: str, bundle: dict[str, bytes], root: Path, force: bool) -> Path:
    skill_md = bundle.get("SKILL.md")
    if skill_md is None:
        raise SystemExit(f"Skill '{name}' has no SKILL.md — cannot convert to Cursor rule.")
    metadata, body = _parse_frontmatter(skill_md.decode())
    description = metadata.get("description", "").replace("\n", " ").strip() or name

    rule_path = root / ".cursor" / "rules" / f"{name}.mdc"
    if rule_path.exists() and not force:
        raise SystemExit(f"Cursor rule already exists at {rule_path}. Use --force to reinstall.")
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    escaped = description.replace("\\", "\\\\").replace('"', '\\"')
    content = "---\n" f'description: "{escaped}"\n' "globs:\n" "alwaysApply: false\n" "---\n" "\n" f"{body}"
    rule_path.write_text(content)
    return rule_path


def _install_agents_md(name: str, bundle: dict[str, bytes], root: Path, force: bool) -> Path:
    skill_md = bundle.get("SKILL.md")
    if skill_md is None:
        raise SystemExit(f"Skill '{name}' has no SKILL.md — cannot inject into AGENTS.md.")
    _, body = _parse_frontmatter(skill_md.decode())

    agents_md = root / "AGENTS.md"
    start_marker = f"<!-- diffusers-skill:{name}:start -->"
    end_marker = f"<!-- diffusers-skill:{name}:end -->"
    section = f"{start_marker}\n\n{body.strip()}\n\n{end_marker}\n"

    existing = agents_md.read_text() if agents_md.exists() else ""
    if start_marker in existing:
        if not force:
            raise SystemExit(f"Skill '{name}' already injected into {agents_md}. Use --force to reinstall.")
        pattern = re.compile(rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}\n?", re.DOTALL)
        new_content = pattern.sub(section, existing)
    else:
        # Ensure exactly one blank line between existing content and the appended section.
        prefix = existing.rstrip("\n") + "\n\n" if existing.strip() else ""
        new_content = prefix + section
    agents_md.write_text(new_content)
    return agents_md


_ADAPTERS = {
    "claude": _install_claude,
    "cursor": _install_cursor,
    "agents-md": _install_agents_md,
}


# ---------------------------------------------------------------------------
# Detection + dispatch
# ---------------------------------------------------------------------------


def _detect_targets() -> tuple[str, ...]:
    """Pick a default target set.

    If an AI agent env var is set, install only to that agent (the agent presumably set up the
    invocation and knows what it needs). Otherwise install to every target so the bundle is
    available in whichever tool the user later switches to.
    """
    for env_var, target in _AGENT_ENV_TO_TARGET.items():
        if os.environ.get(env_var):
            return (target,)
    return _ALL_TARGETS


class SkillsCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "skills",
            help="Install agent skill bundles into Claude, Cursor, and/or AGENTS.md (Codex/Aider).",
            usage="\n  diffusers-cli skills <install|list> [options]",
        )
        parser._optionals.title = "Options"
        actions = parser.add_subparsers(dest="skills_action", required=True, metavar="<action>")

        install = actions.add_parser("install", help="Install a skill bundle from the registry.")
        install.add_argument("name", help="Skill name (e.g. diffusers-cli, custom-blocks).")
        install.add_argument(
            "--agents",
            default=None,
            help=(
                "Comma-separated targets: claude, cursor, agents-md. Defaults to the agent detected "
                "from env vars, or all three if no agent is detected."
            ),
        )
        install.add_argument(
            "--global",
            dest="install_global",
            action="store_true",
            help="Install to $HOME instead of the current project directory.",
        )
        install.add_argument("--force", action="store_true", help="Reinstall over existing files.")
        install.set_defaults(func=SkillsCommand)

        list_action = actions.add_parser("list", help="List available skill bundles in the registry.")
        list_action.set_defaults(func=SkillsCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if self.args.skills_action == "install":
            self._install()
        elif self.args.skills_action == "list":
            self._list()

    def _install(self) -> None:
        targets = self._resolve_targets()
        bundle = _download_bundle(self.args.name)
        root = Path.home() if self.args.install_global else Path.cwd()

        locations = {
            target: str(_ADAPTERS[target](self.args.name, bundle, root, self.args.force))
            for target in targets
        }
        out.result(f"Installed skill '{self.args.name}' to {len(targets)} target(s)", **locations)

    def _resolve_targets(self) -> tuple[str, ...]:
        if not self.args.agents:
            return _detect_targets()
        chosen = tuple(t.strip() for t in self.args.agents.split(",") if t.strip())
        unknown = [t for t in chosen if t not in _ADAPTERS]
        if unknown:
            raise SystemExit(f"Unknown agent target(s): {', '.join(unknown)}. Choose from: {', '.join(_ADAPTERS)}.")
        return chosen

    def _list(self) -> None:
        entries = _fetch_json(_registry_url())
        skills = [{"name": e["name"]} for e in entries if e["type"] == "dir" and not e["name"].startswith(".")]
        if not skills:
            raise SystemExit("No skills found in registry.")
        out.table(skills, headers=["name"])
