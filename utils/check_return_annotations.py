# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team.
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
"""
Check that these methods have a return type annotation:

* `forward()` on every class in `src/diffusers/models`
* `__call__()` on every pipeline in `src/diffusers/pipelines`
* `__call__()` on every modular pipeline block in `src/diffusers/modular_pipelines`

A class counts as a pipeline if it inherits from `DiffusionPipeline`, either directly or through another class. A class
counts as a modular pipeline block if it inherits from `ModularPipelineBlocks` in the same way.

Deprecated code is skipped:

* anything in a folder named `deprecated`, such as `pipelines/deprecated`
* pipelines that inherit from `DeprecatedPipelineMixin`
* classes and methods whose `# Copied from` comment points to deprecated code, because they can't change unless the
  deprecated code changes too

A method is only checked on the class where it's written, not on classes that inherit it. Any annotation passes,
including `-> None`.

Run from the repository root:

    python utils/check_return_annotations.py
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src" / "diffusers"
MODELS_DIR = SRC_DIR / "models"
PIPELINES_DIR = SRC_DIR / "pipelines"
MODULAR_DIR = SRC_DIR / "modular_pipelines"

PIPELINE_BASE = "DiffusionPipeline"
DEPRECATED_PIPELINE_BASE = "DeprecatedPipelineMixin"
BLOCKS_BASE = "ModularPipelineBlocks"


def _base_names(class_def: ast.ClassDef) -> list[str]:
    """Return the names of the classes this class inherits from. For a name like `nn.Module`, keep only `Module`."""
    names = []
    for base in class_def.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _find_method(class_def: ast.ClassDef, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in class_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    return None


def _parse_classes(paths: list[Path]) -> list[tuple[Path, ast.ClassDef, list[str]]]:
    """Return every class in `paths`, along with its file and the lines of that file."""
    classes = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines()
        classes.extend((path, node, lines) for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return classes


def _is_deprecated_path(path: Path) -> bool:
    """Return whether the file is inside a folder named `deprecated`."""
    return "deprecated" in path.relative_to(SRC_DIR).parts[:-1]


def _copied_from_deprecated(lines: list[str], node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether the `# Copied from` comment right above a class or method points to deprecated code."""
    first_line = min([decorator.lineno for decorator in node.decorator_list] + [node.lineno])
    if first_line < 2:
        return False
    comment = lines[first_line - 2].strip()
    return comment.startswith("# Copied from") and ".deprecated." in comment


def _subclass_checker(classes: list[tuple[Path, ast.ClassDef, list[str]]]):
    """
    Return a function `is_subclass(name, base)` that tells whether the class `name` inherits from the class `base`,
    either directly or through other classes.

    Classes are matched by name only, so two classes with the same name in different files are treated as one class.
    """
    bases_by_name: dict[str, set[str]] = defaultdict(set)
    for _, class_def, _ in classes:
        bases_by_name[class_def.name].update(_base_names(class_def))

    cache: dict[tuple[str, str], bool] = {}

    def is_subclass(name: str, base: str, _seen: frozenset[str] = frozenset()) -> bool:
        if name == base:
            return True
        if (name, base) in cache:
            return cache[(name, base)]
        if name in _seen:  # stop if this class was already visited, so a loop in the class names can't run forever
            return False
        result = any(is_subclass(parent, base, _seen | {name}) for parent in bases_by_name.get(name, ()))
        cache[(name, base)] = result
        return result

    return is_subclass


def _is_under(path: Path, directory: Path) -> bool:
    return directory in path.parents


def main() -> int:
    classes = _parse_classes(sorted(SRC_DIR.rglob("*.py")))
    is_subclass = _subclass_checker(classes)

    errors = []
    for path, class_def, lines in classes:
        if _is_deprecated_path(path):
            continue
        if _is_under(path, MODELS_DIR):
            method_name = "forward"
        elif _is_under(path, PIPELINES_DIR):
            if not is_subclass(class_def.name, PIPELINE_BASE) or is_subclass(class_def.name, DEPRECATED_PIPELINE_BASE):
                continue
            method_name = "__call__"
        elif _is_under(path, MODULAR_DIR):
            if not is_subclass(class_def.name, BLOCKS_BASE):
                continue
            method_name = "__call__"
        else:
            continue

        method = _find_method(class_def, method_name)
        if method is None or method.returns is not None:
            continue
        if _copied_from_deprecated(lines, class_def) or _copied_from_deprecated(lines, method):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        errors.append(f"{rel}:{method.lineno}: {class_def.name}.{method_name} has no return type annotation")

    if errors:
        print("\n".join(errors))
        sys.stdout.flush()  # print the list before the summary, even when both end up in the same log
        if len(errors) == 1:
            summary = "Found 1 method without a return type annotation. Add one to the method above."
        else:
            summary = f"Found {len(errors)} methods without a return type annotation. Add one to each method above."
        print(f"\n{summary}", file=sys.stderr)
        return 1

    print("All forward/__call__ methods have return type annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
