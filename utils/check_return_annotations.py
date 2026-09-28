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
* `__call__()` on every pipeline in `src/diffusers/pipelines`, except the ones in `pipelines/deprecated`
* `__call__()` on every modular pipeline block in `src/diffusers/modular_pipelines`

A class counts as a pipeline if it inherits from `DiffusionPipeline`, either directly or through another class. A class
counts as a modular pipeline block if it inherits from `ModularPipelineBlocks` in the same way.

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
DEPRECATED_PIPELINES_DIR = PIPELINES_DIR / "deprecated"

PIPELINE_BASE = "DiffusionPipeline"
BLOCKS_BASE = "ModularPipelineBlocks"

# Classes to skip, written as (file path from the repository root, class name). Only add a class here if its method
# can't be annotated, and add a comment saying why.
IGNORE: set[tuple[str, str]] = {
    # This class is a copy of a class in `pipelines/deprecated` (see its `# Copied from` comment). Annotating it would
    # mean changing the deprecated class too.
    ("src/diffusers/models/unets/unet_stable_cascade.py", "SDCascadeLayerNorm"),
}


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


def _parse_classes(paths: list[Path]) -> list[tuple[Path, ast.ClassDef]]:
    classes = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        classes.extend((path, node) for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return classes


def _subclass_checker(classes: list[tuple[Path, ast.ClassDef]]):
    """
    Return a function `is_subclass(name, base)` that tells whether the class `name` inherits from the class `base`,
    either directly or through other classes.

    Classes are matched by name only, so two classes with the same name in different files are treated as one class.
    """
    bases_by_name: dict[str, set[str]] = defaultdict(set)
    for _, class_def in classes:
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
    for path, class_def in classes:
        if _is_under(path, MODELS_DIR):
            method_name = "forward"
        elif _is_under(path, PIPELINES_DIR) and not _is_under(path, DEPRECATED_PIPELINES_DIR):
            if not is_subclass(class_def.name, PIPELINE_BASE):
                continue
            method_name = "__call__"
        elif _is_under(path, MODULAR_DIR):
            if not is_subclass(class_def.name, BLOCKS_BASE):
                continue
            method_name = "__call__"
        else:
            continue

        rel = path.relative_to(REPO_ROOT).as_posix()
        if (rel, class_def.name) in IGNORE:
            continue
        method = _find_method(class_def, method_name)
        if method is not None and method.returns is None:
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
