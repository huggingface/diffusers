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
import shutil
import sys

import pytest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import modular_auto_docstring  # noqa: E402


@pytest.fixture
def write_file(tmp_path):
    """Write `content` to a file under `tmp_path` and return its path — the entry points all take a filepath."""

    def _write_file(content, name="blocks.py"):
        filepath = os.path.join(tmp_path, name)
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return filepath

    return _write_file


@pytest.fixture
def stub_docs(monkeypatch):
    """
    Generate `Doc for <ClassName>.` instead of importing the file and instantiating the block, so that the tests
    cover the marker/insertion/comparison logic on its own. The `class <ClassName>` first line mirrors what a real
    block's `doc` property returns and is expected to be stripped.
    """
    monkeypatch.setattr(modular_auto_docstring, "load_module", lambda filepath: object())
    monkeypatch.setattr(
        modular_auto_docstring,
        "get_doc_from_class",
        lambda module, class_name: f"class {class_name}\n\nDoc for {class_name}.",
    )


@pytest.mark.parametrize(
    ("filepath", "expected"),
    [
        ("src/diffusers/modular_pipelines/flux/blocks.py", "diffusers.modular_pipelines.flux.blocks"),
        ("./src/diffusers/foo.py", "diffusers.foo"),
        # only a leading `src` is a source root
        ("utils/src/foo.py", "utils.src.foo"),
    ],
)
def test_get_module_from_filepath(filepath, expected):
    assert modular_auto_docstring.get_module_from_filepath(filepath) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # (class_name, class_line, has_existing_docstring, docstring_end_line)
        pytest.param(
            '# auto_docstring\nclass Foo:\n    """old"""\n\n    x = 1\n',
            [("Foo", 2, True, 3)],
            id="existing_docstring",
        ),
        pytest.param("# auto_docstring\nclass Foo:\n    x = 1\n", [("Foo", 2, False, 2)], id="no_docstring"),
        pytest.param(
            '# auto_docstring\nclass Foo:\n    """\n    old\n    """\n\n    x = 1\n',
            [("Foo", 2, True, 5)],
            id="multiline_docstring",
        ),
        pytest.param(
            "# auto_docstring\n\n# another comment\nclass Foo:\n    x = 1\n",
            [("Foo", 4, False, 4)],
            id="marker_separated_by_blanks_and_comments",
        ),
        pytest.param(
            "# auto_docstring\nclass Foo:\n    x = 1\n\n\n# auto_docstring\nclass Bar:\n    x = 1\n",
            [("Foo", 2, False, 2), ("Bar", 7, False, 7)],
            id="multiple_markers_in_file_order",
        ),
        pytest.param(
            "# auto_docstring\nclass Foo:\n    x = 1\n\n\nclass Bar:\n    x = 1\n",
            [("Foo", 2, False, 2)],
            id="unmarked_class_ignored",
        ),
        pytest.param("# auto_docstring\nCONSTANT = 1\n\n\nclass Foo:\n    x = 1\n", [], id="marker_without_class"),
        pytest.param("# auto_docstring for Foo\nclass Foo:\n    x = 1\n", [], id="marker_must_be_whole_comment"),
        pytest.param("# auto_docstring\nclass Foo(:\n", [], id="syntax_error"),
    ],
)
def test_find_auto_docstring_classes(write_file, source, expected):
    assert modular_auto_docstring.find_auto_docstring_classes(write_file(source)) == expected


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        pytest.param("class Foo\n\n\nDoc.", "Doc.", id="strips_class_line_and_blanks"),
        pytest.param("class Bar\n\nDoc.", "class Bar\n\nDoc.", id="keeps_a_different_class_name"),
        pytest.param("class Foo\n\nSee class Foo above.", "See class Foo above.", id="only_strips_first_line"),
    ],
)
def test_strip_class_name_line(doc, expected):
    assert modular_auto_docstring.strip_class_name_line(doc, "Foo") == expected


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        pytest.param("Doc.", '    """Doc."""\n', id="single_line"),
        pytest.param("Doc.\n  Indented.", '    """\n    Doc.\n      Indented.\n    """\n', id="relative_indentation"),
        pytest.param("Doc.\n   \nMore.", '    """\n    Doc.\n\n    More.\n    """\n', id="blank_lines_not_indented"),
        pytest.param("\n\n  Doc.  \n\n", '    """Doc."""\n', id="surrounding_whitespace_stripped"),
    ],
)
def test_format_docstring(doc, expected):
    assert modular_auto_docstring.format_docstring(doc) == expected


def test_format_docstring_custom_indent():
    assert modular_auto_docstring.format_docstring("Doc.", "        ") == '        """Doc."""\n'


def test_class_docstrings_maps_names_to_docstrings():
    docs = modular_auto_docstring._class_docstrings('class Foo:\n    """Doc."""\n\n\nclass Bar:\n    x = 1\n')
    assert docs == {"Foo": "Doc.", "Bar": None}


def test_class_docstrings_are_not_cleaned():
    docs = modular_auto_docstring._class_docstrings('class Foo:\n    """\n    Doc.\n    """\n')
    assert docs["Foo"] == "\n    Doc.\n    "


def build_updated_content(filepath):
    classes = modular_auto_docstring.find_auto_docstring_classes(filepath)
    lines, updated_names = modular_auto_docstring.build_updated_lines(filepath, classes)
    return "".join(lines), updated_names


def test_build_updated_lines_inserts_missing_docstring(write_file, stub_docs):
    filepath = write_file("# auto_docstring\nclass Foo:\n    x = 1\n")
    content, updated_names = build_updated_content(filepath)
    assert content == '# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n    x = 1\n'
    assert updated_names == ["Foo"]


def test_build_updated_lines_replaces_existing_docstring(write_file, stub_docs):
    filepath = write_file('# auto_docstring\nclass Foo:\n    """\n    stale\n    """\n\n    x = 1\n')
    content, _ = build_updated_content(filepath)
    assert content == '# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x = 1\n'


def test_build_updated_lines_strips_class_name_line(write_file, stub_docs):
    filepath = write_file("# auto_docstring\nclass Foo:\n    x = 1\n")
    content, _ = build_updated_content(filepath)
    assert "class Foo\n" not in content.split('"""')[1]


def test_build_updated_lines_keeps_offsets_with_multiple_classes(write_file, stub_docs):
    filepath = write_file(
        '# auto_docstring\nclass Foo:\n    """stale"""\n\n    x = 1\n\n\n# auto_docstring\nclass Bar:\n    y = 2\n'
    )
    content, updated_names = build_updated_content(filepath)
    assert content == (
        '# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x = 1\n\n\n'
        '# auto_docstring\nclass Bar:\n    """Doc for Bar."""\n    y = 2\n'
    )
    # classes are rewritten bottom-up so the earlier line numbers stay valid
    assert updated_names == ["Bar", "Foo"]


def test_build_updated_lines_without_a_generated_doc(write_file, monkeypatch):
    monkeypatch.setattr(modular_auto_docstring, "load_module", lambda filepath: object())
    monkeypatch.setattr(modular_auto_docstring, "get_doc_from_class", lambda module, class_name: None)
    source = '# auto_docstring\nclass Foo:\n    """stale"""\n\n    x = 1\n'
    content, updated_names = build_updated_content(write_file(source))
    assert content == source
    assert updated_names == []


@pytest.fixture
def require_formatters():
    """Check mode regenerates the file through ruff and doc-builder — without them the comparison is meaningless."""
    missing = [tool for tool in ("ruff", "doc-builder") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f'{", ".join(missing)} not found on PATH; install with `pip install -e ".[quality]"`')


@pytest.mark.usefixtures("require_formatters")
class TestProcessFile:
    """Check mode compares the regenerated docstrings against the ones checked into the file."""

    def test_up_to_date_docstring_is_not_reported(self, write_file, stub_docs):
        filepath = write_file('# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x = 1\n')
        assert modular_auto_docstring.process_file(filepath) == []

    def test_stale_docstring_is_reported(self, write_file, stub_docs):
        filepath = write_file('# auto_docstring\nclass Foo:\n    """Doc for Bar."""\n\n    x = 1\n')
        assert modular_auto_docstring.process_file(filepath) == [(filepath, "Foo", 2)]

    def test_missing_docstring_is_reported(self, write_file, stub_docs):
        filepath = write_file("# auto_docstring\nclass Foo:\n    x = 1\n")
        assert modular_auto_docstring.process_file(filepath) == [(filepath, "Foo", 2)]

    def test_unmarked_file_is_not_reported(self, write_file, stub_docs):
        filepath = write_file('class Foo:\n    """stale"""\n\n    x = 1\n')
        assert modular_auto_docstring.process_file(filepath) == []

    def test_only_the_stale_class_is_reported(self, write_file, stub_docs):
        filepath = write_file(
            '# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x = 1\n\n\n'
            '# auto_docstring\nclass Bar:\n    """stale"""\n\n    y = 2\n'
        )
        assert modular_auto_docstring.process_file(filepath) == [(filepath, "Bar", 9)]

    def test_fix_and_overwrite_makes_check_clean(self, write_file, stub_docs):
        filepath = write_file('# auto_docstring\nclass Foo:\n    """stale"""\n\n    x = 1\n')
        modular_auto_docstring.process_file(filepath, overwrite=True)
        with open(filepath, "r", encoding="utf-8", newline="\n") as f:
            assert '"""Doc for Foo."""' in f.read()
        assert modular_auto_docstring.process_file(filepath) == []

    def test_difference_outside_the_docstrings_is_attributed_to_the_first_marked_class(self, write_file, stub_docs):
        # the docstring is current but `x=1` is not ruff-clean, so the regenerated file differs anyway
        filepath = write_file('# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x=1\n')
        assert modular_auto_docstring.process_file(filepath) == [(filepath, "Foo", 2)]


@pytest.mark.usefixtures("require_formatters")
class TestCheckAutoDocstrings:
    """The entry point walks a path, collects every stale class, and reports them together."""

    @pytest.fixture
    def stale_tree(self, tmp_path, write_file):
        """A directory with a stale class at the top level, another one nested, and a file with no markers."""
        write_file('# auto_docstring\nclass Foo:\n    """stale"""\n\n    x = 1\n', name="blocks.py")
        os.makedirs(os.path.join(tmp_path, "sub"))
        write_file('# auto_docstring\nclass Bar:\n    """stale"""\n\n    y = 2\n', name="sub/more_blocks.py")
        write_file("VALUE = 1\n", name="plain.py")
        return str(tmp_path)

    def test_reports_stale_classes_from_every_nested_file(self, stale_tree, stub_docs):
        with pytest.raises(ValueError) as exc_info:
            modular_auto_docstring.check_auto_docstrings(stale_tree)
        message = str(exc_info.value)
        assert f"- {os.path.join(stale_tree, 'blocks.py')}: Foo at line 2" in message
        assert f"- {os.path.join(stale_tree, 'sub', 'more_blocks.py')}: Bar at line 2" in message
        assert "--fix_and_overwrite" in message

    def test_fix_and_overwrite_regenerates_every_nested_file(self, stale_tree, stub_docs):
        modular_auto_docstring.check_auto_docstrings(stale_tree, overwrite=True)
        modular_auto_docstring.check_auto_docstrings(stale_tree)

    def test_up_to_date_tree_does_not_raise(self, tmp_path, write_file, stub_docs):
        write_file('# auto_docstring\nclass Foo:\n    """Doc for Foo."""\n\n    x = 1\n')
        write_file("VALUE = 1\n", name="plain.py")
        modular_auto_docstring.check_auto_docstrings(str(tmp_path))

    def test_accepts_a_single_file(self, write_file, stub_docs):
        filepath = write_file('# auto_docstring\nclass Foo:\n    """stale"""\n\n    x = 1\n')
        with pytest.raises(ValueError, match="Foo at line 2"):
            modular_auto_docstring.check_auto_docstrings(filepath)

    def test_defaults_to_the_diffusers_path(self, stale_tree, stub_docs, monkeypatch):
        monkeypatch.setattr(modular_auto_docstring, "DIFFUSERS_PATH", stale_tree)
        with pytest.raises(ValueError, match="Foo at line 2"):
            modular_auto_docstring.check_auto_docstrings()
