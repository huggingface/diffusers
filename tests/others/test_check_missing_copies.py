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


NORMALIZE_FUNCTION = """def normalize(hidden_states):
    mean_value = hidden_states.mean(dim=-1, keepdim=True)
    variance_value = hidden_states.var(dim=-1, keepdim=True)
    centered = hidden_states - mean_value
    return centered / (variance_value + 1e-5).sqrt()
"""

RENAMED_NORMALIZE_FUNCTION = """def normalize_tokens(tokens):
    token_mean = tokens.mean(dim=-2, keepdim=True)
    token_variance = tokens.var(dim=-2, keepdim=True)
    centered_tokens = tokens - token_mean
    return centered_tokens / (token_variance + 1e-6).sqrt()
"""

UPSAMPLE_CLASS = """class DupUp3D:
    def __init__(self, channels):
        self.channels = channels

    def forward(self, hidden_states):
        hidden_states = hidden_states.repeat_interleave(2, dim=2)
        hidden_states = hidden_states.repeat_interleave(2, dim=3)
        return hidden_states

    def clone(self):
        return DupUp3D(self.channels)
"""


@pytest.fixture
def run_checker(tmp_path, monkeypatch):
    package_dir = tmp_path / "diffusers"
    package_dir.mkdir()
    monkeypatch.setattr(check_missing_copies, "DIFFUSERS_PATH", str(package_dir))

    def check(source_code, copied_code, *, changed_file="copy.py"):
        (package_dir / "source.py").write_text(source_code)
        (package_dir / "copy.py").write_text(copied_code)
        changed_path = package_dir / changed_file
        changed_lines = {str(changed_path): [(1, len(changed_path.read_text().splitlines()))]}
        monkeypatch.setattr(check_missing_copies, "get_changed_lines", lambda base_ref: changed_lines)
        return check_missing_copies.check_missing_copies("base-ref")

    return check


def test_warns_about_exact_copy(run_checker, capsys, tmp_path):
    run_checker(NORMALIZE_FUNCTION, NORMALIZE_FUNCTION)

    warning = capsys.readouterr().out
    copy_path = tmp_path / "diffusers" / "copy.py"
    assert len(warning.splitlines()) == 1
    assert f"::warning file={copy_path},line=1,title=Potential missing Copied from comment::" in warning
    assert "Potential missing # Copied from comment: diffusers.copy.normalize" in warning
    assert "has the same AST as diffusers.source.normalize." in warning


def test_warns_about_copy_with_renamed_variables_and_constants(run_checker, capsys):
    run_checker(NORMALIZE_FUNCTION, RENAMED_NORMALIZE_FUNCTION)

    warning = capsys.readouterr().out
    assert len(warning.splitlines()) == 1
    assert "Potential missing # Copied from comment: diffusers.copy.normalize_tokens" in warning
    assert "has the same AST structure after normalizing names and constants as diffusers.source.normalize." in warning


def test_warns_once_for_copied_class_including_its_methods(run_checker, capsys):
    renamed_class = UPSAMPLE_CLASS.replace("DupUp3D", "QwenImage21DupUp3D")

    run_checker(UPSAMPLE_CLASS, renamed_class)

    warning = capsys.readouterr().out
    assert len(warning.splitlines()) == 1
    assert "Potential missing # Copied from comment: diffusers.copy.QwenImage21DupUp3D has" in warning
    assert "has the same AST structure after normalizing names and constants as diffusers.source.DupUp3D." in warning


def test_no_warning_for_class_with_replacement_annotation(run_checker, capsys):
    renamed_class = UPSAMPLE_CLASS.replace("DupUp3D", "QwenImage21DupUp3D")
    copy_comment = "# Copied from diffusers.source.DupUp3D with DupUp3D->QwenImage21DupUp3D\n"

    missing_copies = run_checker(UPSAMPLE_CLASS, copy_comment + renamed_class)

    assert missing_copies == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "copy_comment",
    [
        pytest.param("# Copied from diffusers.source.normalize\n", id="above-function"),
        pytest.param("# Copied from diffusers.source.normalize\n@staticmethod\n", id="above-decorator"),
        pytest.param("@staticmethod\n# Copied from diffusers.source.normalize\n", id="below-decorator"),
    ],
)
def test_no_warning_for_annotated_function(run_checker, capsys, copy_comment):
    missing_copies = run_checker(NORMALIZE_FUNCTION, copy_comment + NORMALIZE_FUNCTION)

    assert missing_copies == []
    assert capsys.readouterr().out == ""


def test_no_warning_for_short_function(run_checker, capsys):
    identity_function = "def identity(value):\n    return value\n"

    missing_copies = run_checker(identity_function, identity_function)

    assert missing_copies == []
    assert capsys.readouterr().out == ""


def test_no_warning_for_different_function_body(run_checker, capsys):
    different_function = NORMALIZE_FUNCTION.replace("centered /", "centered *")

    missing_copies = run_checker(NORMALIZE_FUNCTION, different_function)

    assert missing_copies == []
    assert capsys.readouterr().out == ""


def test_no_warning_when_referenced_source_changes(run_checker, capsys):
    annotated_copy = "# Copied from diffusers.source.normalize\n" + NORMALIZE_FUNCTION

    missing_copies = run_checker(NORMALIZE_FUNCTION, annotated_copy, changed_file="source.py")

    assert missing_copies == []
    assert capsys.readouterr().out == ""
