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

import os
import sys
from pathlib import Path

import pytest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_ai  # noqa: E402


class TestCheckAiEncoding:
    @pytest.fixture
    def ai_tree(self, tmp_path, monkeypatch):
        """Minimal `.ai/` tree with a UTF-8 curly quote that cp1252 cannot decode."""
        ai_dir = tmp_path / ".ai"
        references = ai_dir / "references"
        skill_dir = ai_dir / "skills" / "demo"
        references.mkdir(parents=True)
        skill_dir.mkdir(parents=True)

        # U+201C LEFT DOUBLE QUOTATION MARK — multi-byte UTF-8, absent from cp1252.
        (references / "guide.md").write_text("See the \u201cattention\u201d notes.\n", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: A demo skill for encoding checks.\n---\n\n"
            "Read references/guide.md for context.\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(check_ai, "AI_DIR", ai_dir)
        monkeypatch.setattr(check_ai, "REFERENCES_DIR", references)
        monkeypatch.setattr(check_ai, "_skill_description", lambda text: "A demo skill for encoding checks.")
        return ai_dir

    def test_main_reads_markdown_as_utf8_under_cp1252_default(self, ai_tree, monkeypatch):
        """Windows locale default (cp1252) must not break guide reads — #14837."""
        real_read_text = Path.read_text

        def locale_default_read_text(self, *args, encoding=None, errors=None, **kwargs):
            if encoding is None and not args:
                # Simulate the Windows non-console default: decode as cp1252.
                return self.read_bytes().decode("cp1252")
            return real_read_text(self, *args, encoding=encoding, errors=errors, **kwargs)

        monkeypatch.setattr(Path, "read_text", locale_default_read_text)

        # Before the fix this raised UnicodeDecodeError on the curly quotes.
        assert check_ai.main() == 0
