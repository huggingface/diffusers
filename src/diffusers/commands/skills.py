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
"""`diffusers-cli skills` — install Agent Skills bundles.

Skill bundles live under `.ai/skills/<name>/` in the diffusers repo and follow the Agent Skills standard: a directory
containing `SKILL.md` (plus optional resources). Installs to `.agents/skills/<name>/` which Claude, Codex, and Cursor
all discover.
"""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

import httpx
from huggingface_hub.cli._output import out

from ..utils import logging
from ..utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from . import BaseDiffusersCLICommand


logger = logging.get_logger("diffusers-cli/skills")


_REGISTRY_BASE = "https://api.github.com/repos/huggingface/diffusers/contents/.ai/skills"
_REGISTRY_REF = "main"

# Native skill-discovery paths per agent. Claude Code reads only `.claude/skills/`; Codex and
# Cursor read `.agents/skills/` (Cursor also honors `.claude/skills/` via compat, but installing
# to `.agents/skills/` is the portable choice for both).
_CLAUDE_SKILLS_DIR = Path(".claude") / "skills"
_AGENTS_SKILLS_DIR = Path(".agents") / "skills"

# Env vars set by each agent when it launches the CLI. Values are the install path to use.
_AGENT_ENV_TO_DIR: dict[str, Path] = {
    "CLAUDECODE": _CLAUDE_SKILLS_DIR,
    "CLAUDE_CODE": _CLAUDE_SKILLS_DIR,
    "CODEX_SANDBOX": _AGENTS_SKILLS_DIR,
    "CURSOR_AI": _AGENTS_SKILLS_DIR,
}
# When no agent env var is set, install to every native path so whichever agent the user
# later switches to picks the skill up.
_ALL_INSTALL_DIRS: tuple[Path, ...] = (_CLAUDE_SKILLS_DIR, _AGENTS_SKILLS_DIR)

# Empty marker dropped inside each installed skill dir so `update` can distinguish our
# installs from user-placed skills at the same paths.
_MANAGED_MARKER_FILE = ".diffusers-skill-managed"


# ---------------------------------------------------------------------------
# Registry fetch
# ---------------------------------------------------------------------------


def _registry_url(name: str = "") -> str:
    """API URL for the registry root, or for a single skill bundle when `name` is given."""
    path = f"/{name}" if name else ""
    return f"{_REGISTRY_BASE}{path}?ref={_REGISTRY_REF}"


def _fetch_json(url: str) -> list[dict]:
    try:
        resp = httpx.get(url, timeout=DIFFUSERS_REQUEST_TIMEOUT)
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


def _download_skill_bundle(name: str) -> dict[str, bytes]:
    files = _walk_skill_files(name)
    if not files:
        raise SystemExit(f"Skill '{name}' has no files in the registry.")
    bundle: dict[str, bytes] = {}
    for rel_path, url in files:
        resp = httpx.get(url, timeout=DIFFUSERS_REQUEST_TIMEOUT)
        resp.raise_for_status()
        bundle[rel_path] = resp.content
    return bundle


# ---------------------------------------------------------------------------
# Install / discovery
# ---------------------------------------------------------------------------


def _detect_install_dirs() -> tuple[Path, ...]:
    """Pick where to install based on the launching agent.

    If we detect a specific agent from its env var, install only there. If nothing is detected, install to every native
    path so any agent picks the skill up later.
    """
    for env_var, skills_dir in _AGENT_ENV_TO_DIR.items():
        if os.environ.get(env_var):
            return (skills_dir,)
    return _ALL_INSTALL_DIRS


def _install_skill(name: str, bundle: dict[str, bytes], root: Path, skills_dir: Path, force: bool) -> Path:
    skill_dir = root / skills_dir / name
    if skill_dir.exists():
        if not force:
            raise SystemExit(f"Skill already installed at {skill_dir}. Use --force to reinstall.")
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, data in bundle.items():
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (skill_dir / _MANAGED_MARKER_FILE).touch()
    return skill_dir


def _has_local_changes(skill_dir: Path, bundle: dict[str, bytes]) -> bool:
    """True if the installed skill has any file that differs from `bundle` or has extra files.

    The marker file is ignored. Compares raw bytes so a whitespace-only edit still counts as dirty.
    """
    on_disk: dict[str, bytes] = {}
    for path in skill_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(skill_dir))
        if rel == _MANAGED_MARKER_FILE:
            continue
        on_disk[rel] = path.read_bytes()
    return on_disk != bundle


def _discover_installed(root: Path) -> list[tuple[Path, str]]:
    """Return `(skills_dir, name)` pairs for every managed install under `root`."""
    found: list[tuple[Path, str]] = []
    for skills_dir in _ALL_INSTALL_DIRS:
        skills_root = root / skills_dir
        if not skills_root.exists():
            continue
        for d in sorted(skills_root.iterdir()):
            if d.is_dir() and (d / _MANAGED_MARKER_FILE).exists():
                found.append((skills_dir, d.name))
    return found


class SkillsCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "skills",
            help="Manage Agent Skills for AI assistants.",
            usage="\n  diffusers-cli skills <add|list|update|preview> [options]",
        )
        parser._optionals.title = "Options"
        actions = parser.add_subparsers(dest="skills_action", required=True, metavar="<action>")

        add = actions.add_parser("add", help="Download and install a skill.")
        add.add_argument(
            "name",
            nargs="?",
            default=None,
            help="Skill name (e.g. diffusers-cli, custom-blocks). Omit and pass --all to install every skill.",
        )
        add.add_argument(
            "--all",
            dest="install_all",
            action="store_true",
            help="Install every skill in the registry. Mutually exclusive with a positional name.",
        )
        add.add_argument(
            "--global",
            "-g",
            dest="install_global",
            action="store_true",
            help="Install globally (user-level) instead of in the current project directory.",
        )
        add.add_argument("--force", action="store_true", help="Overwrite existing skills in the destination.")
        add.set_defaults(func=SkillsCommand)

        list_action = actions.add_parser("list", help="List available skills in the registry.")
        list_action.set_defaults(func=SkillsCommand)

        update = actions.add_parser("update", help="Re-download and reinstall managed skills.")
        update.add_argument(
            "name",
            nargs="?",
            default=None,
            help="Optional installed skill name to update. Omit to update every managed skill.",
        )
        update.add_argument(
            "--global",
            "-g",
            dest="install_global",
            action="store_true",
            help="Update skills installed globally (user-level) instead of the current project.",
        )
        update.add_argument(
            "--force",
            action="store_true",
            help="Overwrite skills even if they have local modifications since install.",
        )
        update.set_defaults(func=SkillsCommand)

        preview = actions.add_parser("preview", help="Print a skill's SKILL.md from the registry.")
        preview.add_argument("name", help="Skill name to preview.")
        preview.set_defaults(func=SkillsCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        if self.args.skills_action == "add":
            self._add()
        elif self.args.skills_action == "list":
            self._list()
        elif self.args.skills_action == "update":
            self._update()
        elif self.args.skills_action == "preview":
            self._preview()

    def _add(self) -> None:
        if self.args.install_all and self.args.name:
            raise SystemExit("--all and a positional skill name are mutually exclusive.")
        if not self.args.install_all and not self.args.name:
            raise SystemExit("Pass a skill name (e.g. diffusers-cli) or --all to install every skill.")

        root = Path.home() if self.args.install_global else Path.cwd()
        install_dirs = _detect_install_dirs()
        names = self._resolve_names()

        installed: list[str] = []
        failed: list[str] = []
        for name in names:
            try:
                bundle = _download_skill_bundle(name)
                for skills_dir in install_dirs:
                    _install_skill(name, bundle, root, skills_dir, self.args.force)
                installed.append(name)
            except (SystemExit, httpx.HTTPError) as e:
                # Downgrade to a warning so one broken skill doesn't abort the batch.
                logger.warning(f"Skipping skill {name!r}: {e}")
                failed.append(name)

        if not installed:
            raise SystemExit(f"No skills installed. Failed: {failed}")
        out.result(
            f"Installed {len(installed)} skill(s)",
            installed=", ".join(installed),
            failed=", ".join(failed) if failed else None,
            paths=", ".join(str(root / d) for d in install_dirs),
        )

    def _update(self) -> None:
        root = Path.home() if self.args.install_global else Path.cwd()
        installed = _discover_installed(root)
        if self.args.name is not None:
            installed = [entry for entry in installed if entry[1] == self.args.name]
            if not installed:
                raise SystemExit(f"No installed skill named {self.args.name!r} found under {root}.")
        if not installed:
            raise SystemExit(f"No managed skills found under {root}.")

        # Group by skill name so we redownload each bundle once even if it's installed to
        # multiple locations (e.g. both .claude/skills/ and .agents/skills/).
        by_name: dict[str, list[Path]] = {}
        for skills_dir, name in installed:
            by_name.setdefault(name, []).append(skills_dir)

        updated: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        for name, dirs in sorted(by_name.items()):
            try:
                bundle = _download_skill_bundle(name)
                for skills_dir in dirs:
                    skill_dir = root / skills_dir / name
                    if not self.args.force and _has_local_changes(skill_dir, bundle):
                        logger.warning(
                            f"Skill {name!r} at {skill_dir} has local modifications; "
                            "skipping. Pass --force to overwrite them."
                        )
                        skipped.append(name)
                        continue
                    _install_skill(name, bundle, root, skills_dir, force=True)
                    updated.append(name)
            except (SystemExit, httpx.HTTPError) as e:
                logger.warning(f"Skipping skill {name!r}: {e}")
                failed.append(name)

        out.result(
            f"Updated {len(updated)} skill(s)",
            updated=", ".join(updated),
            skipped=", ".join(skipped) if skipped else None,
            failed=", ".join(failed) if failed else None,
        )

    def _preview(self) -> None:
        bundle = _download_skill_bundle(self.args.name)
        skill_md = bundle.get("SKILL.md")
        if skill_md is None:
            raise SystemExit(f"Skill {self.args.name!r} has no SKILL.md in the registry.")
        print(skill_md.decode())

    def _list(self) -> None:
        entries = _fetch_json(_registry_url())
        skills = [{"name": e["name"]} for e in entries if e["type"] == "dir" and not e["name"].startswith(".")]
        if not skills:
            raise SystemExit("No skills found in registry.")
        out.table(skills, headers=["name"])

    def _resolve_names(self) -> list[str]:
        if self.args.install_all:
            entries = _fetch_json(_registry_url())
            return sorted(e["name"] for e in entries if e["type"] == "dir" and not e["name"].startswith("."))
        return [self.args.name]
