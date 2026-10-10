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

import argparse
import ast
import glob
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass


DIFFUSERS_PATH = "src/diffusers"
MIN_AST_NODES = 20

_re_copy_warning = re.compile(r"^\s*#\s*Copied from\s+diffusers\.(\S+\.\S+)(?:\s|$)")
_re_diff_file = re.compile(r"^\+\+\+ b/(.+)$")
_re_diff_hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Definition:
    path: str
    qualified_name: str
    line: int
    end_line: int
    kind: str
    fingerprint: str
    normalized_fingerprint: str
    ast_nodes: int
    copied_from: str | None
    covered_by_copy: bool


@dataclass(frozen=True)
class MissingCopy:
    definition: Definition
    source: str
    exact: bool


class DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.definitions = []
        self.name_stack = []
        self.copy_stack = []

    def visit_ClassDef(self, node):
        self._visit_definition(node)

    def visit_FunctionDef(self, node):
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_definition(node)

    def _visit_definition(self, node):
        copied_from = _find_copy_comment(node, self.lines)
        body = ast.Module(body=node.body, type_ignores=[])
        fingerprint = ast.dump(body, include_attributes=False)
        module_name = _module_name(self.path)
        qualified_name = ".".join([module_name, *self.name_stack, node.name])
        self.definitions.append(
            Definition(
                path=self.path,
                qualified_name=qualified_name,
                line=node.lineno,
                end_line=node.end_lineno,
                kind=type(node).__name__,
                fingerprint=fingerprint,
                normalized_fingerprint=_normalized_ast_dump(body),
                ast_nodes=sum(1 for _ in ast.walk(body)),
                copied_from=copied_from,
                covered_by_copy=copied_from is not None or any(self.copy_stack),
            )
        )
        self.name_stack.append(node.name)
        self.copy_stack.append(copied_from is not None)
        self.generic_visit(node)
        self.copy_stack.pop()
        self.name_stack.pop()


def _module_name(path):
    relative_path = os.path.relpath(path, os.path.dirname(DIFFUSERS_PATH))
    module_name = os.path.splitext(relative_path)[0].replace(os.sep, ".")
    return module_name.removesuffix(".__init__")


def _normalized_ast_dump(node):
    values = {}
    counts = Counter()

    def normalize(value, constant_value=False):
        if isinstance(value, ast.AST):
            return (
                type(value).__name__,
                tuple(
                    (field, normalize(child, isinstance(value, ast.Constant) and field == "value"))
                    for field, child in ast.iter_fields(value)
                ),
            )
        if isinstance(value, list):
            return tuple(normalize(item) for item in value)
        if isinstance(value, str) or (
            constant_value and isinstance(value, (int, float, complex)) and not isinstance(value, bool)
        ):
            key = (type(value), value)
            if key not in values:
                value_type = "string" if isinstance(value, str) else "number"
                values[key] = (value_type, counts[value_type])
                counts[value_type] += 1
            return values[key]
        return value

    return repr(normalize(node))


def _find_copy_comment(node, lines):
    first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    for line_index in range(first_line - 1, node.lineno - 1):
        match = _re_copy_warning.match(lines[line_index])
        if match is not None:
            return match.group(1)

    line_index = first_line - 2
    while line_index >= 0 and lines[line_index].lstrip().startswith("#"):
        match = _re_copy_warning.match(lines[line_index])
        if match is not None:
            return match.group(1)
        line_index -= 1
    return None


def collect_definitions():
    definitions = []
    for path in glob.glob(os.path.join(DIFFUSERS_PATH, "**/*.py"), recursive=True):
        with open(path, encoding="utf-8") as source_file:
            source = source_file.read()
        visitor = DefinitionVisitor(path, source.splitlines())
        visitor.visit(ast.parse(source, filename=path))
        definitions.extend(visitor.definitions)
    return definitions


def parse_changed_lines(diff):
    changed_lines = defaultdict(list)
    current_path = None
    for line in diff.splitlines():
        file_match = _re_diff_file.match(line)
        if file_match is not None:
            current_path = file_match.group(1)
            continue
        hunk_match = _re_diff_hunk.match(line)
        if hunk_match is None or current_path is None:
            continue
        start = int(hunk_match.group(1))
        count = int(hunk_match.group(2) or 1)
        if count > 0:
            changed_lines[current_path].append((start, start + count - 1))
    return dict(changed_lines)


def get_changed_lines(base_ref):
    diff = subprocess.run(
        ["git", "diff", "--unified=0", "--diff-filter=ACMR", base_ref, "HEAD", "--", DIFFUSERS_PATH],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_changed_lines(diff)


def _is_changed(definition, changed_lines):
    return any(
        definition.line <= changed_end and changed_start <= definition.end_line
        for changed_start, changed_end in changed_lines.get(definition.path, [])
    )


def _source_priority(definition, changed_lines):
    return (
        _is_changed(definition, changed_lines),
        definition.copied_from is not None,
        definition.path,
        definition.line,
    )


def find_missing_copies(changed_lines, min_ast_nodes=MIN_AST_NODES):
    definitions = collect_definitions()
    definitions_by_fingerprint = defaultdict(list)
    for definition in definitions:
        if definition.ast_nodes >= min_ast_nodes:
            definitions_by_fingerprint[(definition.kind, definition.normalized_fingerprint)].append(definition)

    changed_definitions = sorted(
        (definition for definition in definitions if _is_changed(definition, changed_lines)),
        key=lambda definition: (definition.path, definition.line, -definition.end_line),
    )
    duplicate_ancestors = defaultdict(list)
    missing_copies = []
    for definition in changed_definitions:
        if definition.covered_by_copy or definition.ast_nodes < min_ast_nodes:
            continue
        if any(
            start <= definition.line and definition.end_line <= end
            for start, end in duplicate_ancestors[definition.path]
        ):
            continue

        matches = [
            match
            for match in definitions_by_fingerprint[(definition.kind, definition.normalized_fingerprint)]
            if (match.path, match.line) != (definition.path, definition.line)
        ]
        if not matches:
            continue

        duplicate_ancestors[definition.path].append((definition.line, definition.end_line))
        if any(match.copied_from == definition.qualified_name.removeprefix("diffusers.") for match in matches):
            continue

        source = min(matches, key=lambda match: _source_priority(match, changed_lines))
        if _is_changed(source, changed_lines) and (definition.path, definition.line) < (source.path, source.line):
            continue
        source_name = f"diffusers.{source.copied_from}" if source.copied_from else source.qualified_name
        missing_copies.append(
            MissingCopy(definition=definition, source=source_name, exact=definition.fingerprint == source.fingerprint)
        )

    return missing_copies


def check_missing_copies(base_ref):
    missing_copies = find_missing_copies(get_changed_lines(base_ref))
    for missing_copy in missing_copies:
        definition = missing_copy.definition
        match_kind = "same AST" if missing_copy.exact else "same AST structure after normalizing names and constants"
        print(
            f"::warning file={definition.path},line={definition.line},title=Potential missing Copied from comment::"
            f"Potential missing # Copied from comment: {definition.qualified_name} has the {match_kind} as "
            f"{missing_copy.source}."
        )
    return missing_copies


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("base_ref")
    args = parser.parse_args(argv)
    return 1 if check_missing_copies(args.base_ref) else 0


if __name__ == "__main__":
    raise SystemExit(main())
