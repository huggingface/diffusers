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
"""Unit tests for ``diffusers.commands.skills``.

Exercises the install / discovery / agent-detection logic. The GitHub Contents API path
(``_fetch_json`` / ``_download_skill_bundle``) is left out — it needs the network.
"""

import pytest

from diffusers.commands.skills import (
    _AGENT_ENV_TO_DIR,
    _AGENTS_SKILLS_DIR,
    _ALL_INSTALL_DIRS,
    _CLAUDE_SKILLS_DIR,
    _MANAGED_MARKER_FILE,
    _detect_install_dirs,
    _discover_installed,
    _install_skill,
)


SAMPLE_BUNDLE = {
    "SKILL.md": b"---\nname: example-skill\ndescription: An example skill.\n---\n\n# Example\n",
    "extra.md": b"# Extra reference file\n",
    "reference/script.py": b"print('hello from a nested reference')\n",
}


class TestInstallSkill:
    def test_writes_bundle_and_marker(self, tmp_path):
        location = _install_skill("example-skill", SAMPLE_BUNDLE, tmp_path, _CLAUDE_SKILLS_DIR, force=False)
        skill_dir = tmp_path / _CLAUDE_SKILLS_DIR / "example-skill"
        assert location == skill_dir
        assert (skill_dir / "SKILL.md").read_bytes() == SAMPLE_BUNDLE["SKILL.md"]
        assert (skill_dir / "extra.md").read_bytes() == SAMPLE_BUNDLE["extra.md"]
        assert (skill_dir / "reference" / "script.py").read_bytes() == SAMPLE_BUNDLE["reference/script.py"]
        assert (skill_dir / _MANAGED_MARKER_FILE).exists()

    def test_errors_without_force_on_existing_dir(self, tmp_path):
        _install_skill("example-skill", SAMPLE_BUNDLE, tmp_path, _CLAUDE_SKILLS_DIR, force=False)
        with pytest.raises(SystemExit, match="Use --force to reinstall"):
            _install_skill("example-skill", SAMPLE_BUNDLE, tmp_path, _CLAUDE_SKILLS_DIR, force=False)

    def test_force_replaces_existing(self, tmp_path):
        _install_skill("example-skill", SAMPLE_BUNDLE, tmp_path, _CLAUDE_SKILLS_DIR, force=False)
        new_bundle = {"SKILL.md": b"different content"}
        _install_skill("example-skill", new_bundle, tmp_path, _CLAUDE_SKILLS_DIR, force=True)
        skill_dir = tmp_path / _CLAUDE_SKILLS_DIR / "example-skill"
        assert (skill_dir / "SKILL.md").read_bytes() == b"different content"
        assert not (skill_dir / "extra.md").exists()
        assert (skill_dir / _MANAGED_MARKER_FILE).exists()


class TestDetectInstallDirs:
    def test_no_env_installs_everywhere(self, monkeypatch):
        for var in _AGENT_ENV_TO_DIR:
            monkeypatch.delenv(var, raising=False)
        assert _detect_install_dirs() == _ALL_INSTALL_DIRS

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("CLAUDECODE", _CLAUDE_SKILLS_DIR),
            ("CLAUDE_CODE", _CLAUDE_SKILLS_DIR),
            ("CODEX_SANDBOX", _AGENTS_SKILLS_DIR),
            ("CURSOR_AI", _AGENTS_SKILLS_DIR),
        ],
    )
    def test_env_var_picks_specific_target(self, monkeypatch, env_var, expected):
        for var in _AGENT_ENV_TO_DIR:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(env_var, "1")
        assert _detect_install_dirs() == (expected,)


class TestDiscoverInstalled:
    def test_returns_managed_installs_from_all_paths(self, tmp_path):
        _install_skill("skill-a", SAMPLE_BUNDLE, tmp_path, _CLAUDE_SKILLS_DIR, force=False)
        _install_skill("skill-b", SAMPLE_BUNDLE, tmp_path, _AGENTS_SKILLS_DIR, force=False)
        # User-placed skill without our marker — should be ignored.
        unmanaged = tmp_path / _CLAUDE_SKILLS_DIR / "user-placed"
        unmanaged.mkdir(parents=True)
        (unmanaged / "SKILL.md").write_bytes(b"user content")

        assert _discover_installed(tmp_path) == [
            (_CLAUDE_SKILLS_DIR, "skill-a"),
            (_AGENTS_SKILLS_DIR, "skill-b"),
        ]

    def test_returns_empty_when_no_skills_dir(self, tmp_path):
        assert _discover_installed(tmp_path) == []
