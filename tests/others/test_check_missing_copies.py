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

import pytest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_missing_copies  # noqa: E402


SOURCE = """def source(value):
    total = value + 1
    total = total * 2
    total = total - 3
    if total > 0:
        return total
    return -total
"""

COPY = """def copied(value):
    total = value + 1
    total = total * 2
    total = total - 3
    if total > 0:
        return total
    return -total
"""

RENAMED_COPY = """def copied(sample):
    result = sample + 4
    result = result * 5
    result = result - 6
    if result > 0:
        return result
    return -result
"""

WITH_REPLACEMENT_SOURCE = """class DupUp3D:
    def __init__(self, channels):
        self.channels = channels

    def clone(self):
        return DupUp3D(self.channels)
"""

WITH_REPLACEMENT_COPY = WITH_REPLACEMENT_SOURCE.replace("DupUp3D", "QwenImage21DupUp3D")


@pytest.fixture
def diffusers_dir(tmp_path, monkeypatch):
    package_dir = tmp_path / "diffusers"
    package_dir.mkdir()
    monkeypatch.setattr(check_missing_copies, "DIFFUSERS_PATH", str(package_dir))
    return package_dir


def test_parse_changed_lines():
    diff = """diff --git a/src/diffusers/source.py b/src/diffusers/source.py
--- a/src/diffusers/source.py
+++ b/src/diffusers/source.py
@@ -1 +1,2 @@
@@ -8,2 +9 @@
diff --git a/src/diffusers/deleted.py b/src/diffusers/deleted.py
--- a/src/diffusers/deleted.py
+++ /dev/null
@@ -1,3 +0,0 @@
"""

    assert check_missing_copies.parse_changed_lines(diff) == {"src/diffusers/source.py": [(1, 2), (9, 9)]}


def test_reports_changed_duplicate(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text(SOURCE)
    copy_path.write_text(COPY)

    missing_copies = check_missing_copies.find_missing_copies(
        {str(copy_path): [(1, len(COPY.splitlines()))]}, min_ast_nodes=1
    )

    assert len(missing_copies) == 1
    assert missing_copies[0].definition.qualified_name == "diffusers.copy.copied"
    assert missing_copies[0].source == "diffusers.source.source"
    assert missing_copies[0].exact


def test_reports_duplicate_with_systematic_replacements(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text(SOURCE)
    copy_path.write_text(RENAMED_COPY)

    missing_copies = check_missing_copies.find_missing_copies(
        {str(copy_path): [(1, len(RENAMED_COPY.splitlines()))]}, min_ast_nodes=1
    )

    assert len(missing_copies) == 1
    assert missing_copies[0].definition.qualified_name == "diffusers.copy.copied"
    assert missing_copies[0].source == "diffusers.source.source"
    assert not missing_copies[0].exact


def test_with_based_replacement(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text(WITH_REPLACEMENT_SOURCE)
    copy_path.write_text(WITH_REPLACEMENT_COPY)
    changed_lines = {str(copy_path): [(1, len(WITH_REPLACEMENT_COPY.splitlines()))]}

    missing_copies = check_missing_copies.find_missing_copies(changed_lines, min_ast_nodes=1)

    assert len(missing_copies) == 1
    assert missing_copies[0].definition.qualified_name == "diffusers.copy.QwenImage21DupUp3D"
    assert missing_copies[0].source == "diffusers.source.DupUp3D"
    assert not missing_copies[0].exact

    copy_comment = "# Copied from diffusers.source.DupUp3D with DupUp3D->QwenImage21DupUp3D\n"
    copy_path.write_text(copy_comment + WITH_REPLACEMENT_COPY)
    changed_lines = {str(copy_path): [(1, len((copy_comment + WITH_REPLACEMENT_COPY).splitlines()))]}

    assert check_missing_copies.find_missing_copies(changed_lines, min_ast_nodes=1) == []


@pytest.mark.parametrize(
    "copy_header",
    [
        "# Copied from diffusers.source.source\n",
        "# Copied from diffusers.source.source\n@staticmethod\n",
        "@staticmethod\n# Copied from diffusers.source.source\n",
    ],
)
def test_ignores_annotated_duplicate(diffusers_dir, copy_header):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text(SOURCE)
    copy_path.write_text(copy_header + COPY)

    missing_copies = check_missing_copies.find_missing_copies(
        {str(copy_path): [(1, len((copy_header + COPY).splitlines()))]}, min_ast_nodes=1
    )

    assert missing_copies == []


def test_ignores_short_duplicate(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text("def source(value):\n    return value\n")
    copy_path.write_text("def copied(value):\n    return value\n")

    missing_copies = check_missing_copies.find_missing_copies({str(copy_path): [(1, 2)]}, min_ast_nodes=20)

    assert missing_copies == []


def test_reports_only_enclosing_duplicate(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text("class Source:\n" + "\n".join(f"    {line}" for line in SOURCE.splitlines()) + "\n")
    copy_path.write_text("class Copy:\n" + "\n".join(f"    {line}" for line in SOURCE.splitlines()) + "\n")

    missing_copies = check_missing_copies.find_missing_copies(
        {str(copy_path): [(1, len(COPY.splitlines()) + 1)]}, min_ast_nodes=1
    )

    assert len(missing_copies) == 1
    assert missing_copies[0].definition.qualified_name == "diffusers.copy.Copy"


def test_does_not_report_the_referenced_source(diffusers_dir):
    source_path = diffusers_dir / "source.py"
    copy_path = diffusers_dir / "copy.py"
    source_path.write_text(SOURCE)
    copy_path.write_text("# Copied from diffusers.source.source\n" + COPY)

    missing_copies = check_missing_copies.find_missing_copies(
        {str(source_path): [(1, len(SOURCE.splitlines()))]}, min_ast_nodes=1
    )

    assert missing_copies == []


@pytest.mark.parametrize(("missing_copies", "expected_status"), [([], 0), ([object()], 1)])
def test_main_exit_status(monkeypatch, missing_copies, expected_status):
    monkeypatch.setattr(check_missing_copies, "check_missing_copies", lambda base_ref: missing_copies)

    assert check_missing_copies.main(["base-ref"]) == expected_status
