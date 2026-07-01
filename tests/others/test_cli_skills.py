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

Covers the pure-transform pieces (frontmatter parsing, format adapters, target detection) without
hitting the network. The GitHub Contents API path (``_fetch_json``/``_download_bundle``) is left
out — exercise it via an integration smoke test if it ever regresses.
"""

import pytest

from diffusers.commands.skills import (
    _AGENT_ENV_TO_TARGET,
    _ALL_TARGETS,
    _detect_targets,
    _install_agents_md,
    _install_claude,
    _install_cursor,
    _parse_frontmatter,
)


SAMPLE_SKILL_MD = b"""---
name: example-skill
description: >
  Example skill body - multi-line description gets joined
  into a single string by the parser.
---

# Example Skill

Body content lives here.
"""

SAMPLE_BUNDLE = {
    "SKILL.md": SAMPLE_SKILL_MD,
    "extra.md": b"# Extra reference file\n\nSupplementary content.\n",
}


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_no_frontmatter_returns_empty_metadata(self):
        text = "# Just markdown\n\nNo frontmatter here."
        metadata, body = _parse_frontmatter(text)
        assert metadata == {}
        assert body == text

    def test_simple_key_value(self):
        text = "---\nname: foo\ndescription: bar baz\n---\n\nbody"
        metadata, body = _parse_frontmatter(text)
        assert metadata == {"name": "foo", "description": "bar baz"}
        assert body == "\nbody"

    def test_block_scalar_description(self):
        text = SAMPLE_SKILL_MD.decode()
        metadata, body = _parse_frontmatter(text)
        assert metadata["name"] == "example-skill"
        assert "multi-line description gets joined" in metadata["description"]
        assert "\n" not in metadata["description"]  # joined into one line
        assert body.startswith("\n# Example Skill")

    def test_malformed_frontmatter_missing_close(self):
        text = "---\nname: foo\n\nbody without closing fence"
        metadata, body = _parse_frontmatter(text)
        assert metadata == {}
        assert body == text


# ---------------------------------------------------------------------------
# _install_claude
# ---------------------------------------------------------------------------


class TestInstallClaude:
    def test_copies_bundle_verbatim(self, tmp_path):
        location = _install_claude("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        skill_dir = tmp_path / ".claude" / "skills" / "example-skill"
        assert location == skill_dir
        assert (skill_dir / "SKILL.md").read_bytes() == SAMPLE_SKILL_MD
        assert (skill_dir / "extra.md").exists()

    def test_errors_without_force_on_existing_dir(self, tmp_path):
        _install_claude("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        with pytest.raises(SystemExit, match="Use --force to reinstall"):
            _install_claude("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)

    def test_force_replaces_existing(self, tmp_path):
        _install_claude("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        # Mutate one file then reinstall with new content
        new_bundle = {"SKILL.md": b"different content"}
        _install_claude("example-skill", new_bundle, tmp_path, force=True)
        skill_dir = tmp_path / ".claude" / "skills" / "example-skill"
        assert (skill_dir / "SKILL.md").read_bytes() == b"different content"
        # Files from the old install are gone
        assert not (skill_dir / "extra.md").exists()


# ---------------------------------------------------------------------------
# _install_cursor
# ---------------------------------------------------------------------------


class TestInstallCursor:
    def test_writes_mdc_with_cursor_frontmatter(self, tmp_path):
        location = _install_cursor("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        assert location == tmp_path / ".cursor" / "rules" / "example-skill.mdc"
        content = location.read_text()
        # Cursor frontmatter shape, not Claude's
        assert content.startswith("---\n")
        assert 'description: "' in content
        assert "globs:" in content
        assert "alwaysApply: false" in content
        # Body preserved
        assert "# Example Skill" in content

    def test_escapes_quotes_and_backslashes_in_description(self, tmp_path):
        bundle = {
            "SKILL.md": b'---\nname: x\ndescription: She said "hello" with \\backslash.\n---\n\nbody',
        }
        _install_cursor("x", bundle, tmp_path, force=False)
        content = (tmp_path / ".cursor" / "rules" / "x.mdc").read_text()
        # Quotes and backslashes escaped inside the YAML string
        assert r'\"hello\"' in content
        assert r"\\backslash" in content

    def test_errors_without_skill_md(self, tmp_path):
        with pytest.raises(SystemExit, match="no SKILL.md"):
            _install_cursor("x", {"other.md": b"no skill.md here"}, tmp_path, force=False)

    def test_errors_without_force_on_existing(self, tmp_path):
        _install_cursor("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        with pytest.raises(SystemExit, match="already exists"):
            _install_cursor("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)


# ---------------------------------------------------------------------------
# _install_agents_md
# ---------------------------------------------------------------------------


class TestInstallAgentsMd:
    def test_creates_agents_md_when_missing(self, tmp_path):
        location = _install_agents_md("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        assert location == tmp_path / "AGENTS.md"
        content = location.read_text()
        assert "<!-- diffusers-skill:example-skill:start -->" in content
        assert "<!-- diffusers-skill:example-skill:end -->" in content
        assert "# Example Skill" in content

    def test_appends_to_existing_agents_md(self, tmp_path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Pre-existing project conventions\n\nDon't touch this section.\n")
        _install_agents_md("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        content = agents_md.read_text()
        assert "Pre-existing project conventions" in content
        assert "Don't touch this section." in content
        assert "<!-- diffusers-skill:example-skill:start -->" in content
        # Section appended after the existing content, separated by blank line
        assert content.index("Don't touch") < content.index("diffusers-skill")

    def test_force_replaces_existing_section_in_place(self, tmp_path):
        _install_agents_md("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        new_bundle = {
            "SKILL.md": b"---\nname: example-skill\ndescription: x\n---\n\n# Updated body\n",
        }
        _install_agents_md("example-skill", new_bundle, tmp_path, force=True)
        content = (tmp_path / "AGENTS.md").read_text()
        assert "# Updated body" in content
        assert "# Example Skill" not in content
        # Markers exactly once each
        assert content.count("<!-- diffusers-skill:example-skill:start -->") == 1
        assert content.count("<!-- diffusers-skill:example-skill:end -->") == 1

    def test_errors_without_force_on_existing_section(self, tmp_path):
        _install_agents_md("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)
        with pytest.raises(SystemExit, match="already injected"):
            _install_agents_md("example-skill", SAMPLE_BUNDLE, tmp_path, force=False)

    def test_multiple_skills_coexist(self, tmp_path):
        _install_agents_md("skill-a", SAMPLE_BUNDLE, tmp_path, force=False)
        bundle_b = {"SKILL.md": b"---\nname: skill-b\ndescription: b\n---\n\n# B body\n"}
        _install_agents_md("skill-b", bundle_b, tmp_path, force=False)
        content = (tmp_path / "AGENTS.md").read_text()
        assert "<!-- diffusers-skill:skill-a:start -->" in content
        assert "<!-- diffusers-skill:skill-b:start -->" in content


# ---------------------------------------------------------------------------
# _detect_targets
# ---------------------------------------------------------------------------


class TestDetectTargets:
    def test_no_env_vars_returns_all_targets(self, monkeypatch):
        for var in _AGENT_ENV_TO_TARGET:
            monkeypatch.delenv(var, raising=False)
        assert _detect_targets() == _ALL_TARGETS

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("CLAUDECODE", "claude"),
            ("CLAUDE_CODE", "claude"),
            ("CURSOR_AI", "cursor"),
            ("CODEX_SANDBOX", "agents-md"),
            ("AIDER_AI_CONTEXT", "agents-md"),
        ],
    )
    def test_env_var_picks_single_target(self, monkeypatch, env_var, expected):
        for var in _AGENT_ENV_TO_TARGET:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(env_var, "1")
        assert _detect_targets() == (expected,)
